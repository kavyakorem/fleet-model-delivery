"use strict";
const el = id => document.getElementById(id);
const ids = ["canary", "replay", "promote", "rollback"];
let busy = false;
let state;
const fmt = (v, digits = 3) => v == null ? "—" : Math.abs(v) < .0001 && v !== 0 ? Number(v).toExponential(2) : Number(v).toFixed(digits);

function table(id, data, styleColumn = -1) {
  const body = el(id);
  body.replaceChildren();
  for (const row of data) {
    const tr = document.createElement("tr");
    row.forEach((value, index) => {
      const td = document.createElement("td");
      td.textContent = value;
      if (index === styleColumn) td.className = /^(pass|normal)$/i.test(value) ? "pass" : "warn";
      tr.append(td);
    });
    body.append(tr);
  }
}
function controls() {
  ids.forEach(id => { el(id).disabled = busy || !state; });
  if (!state || busy) return;
  el("canary").disabled = !!state.candidate;
  el("promote").disabled = !state.candidate;
  el("rollback").disabled = !state.candidate && !state.previous;
}
async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : data.detail?.reason || "Request failed");
  return data;
}
async function refresh() {
  const data = await request("/api/status");
  state = data.state;
  el("active").textContent = state.active;
  el("candidate").textContent = state.candidate || "—";
  el("share").textContent = state.candidate ? state.percent + "% assigned by window hash" : "No canary running";
  el("total").textContent = data.total.toLocaleString();
  el("alerts").textContent = data.alerts.toLocaleString();
  el("trial").replaceChildren();
  for (const [key, value] of [["Canary windows", data.trial.windows], ["Labeled windows", data.trial.labeled], ["Labeled faults", data.trial.faults], ["Canary F1", fmt(data.trial.f1)], ["Canary inference p95", fmt(data.trial.p95_ms) + " ms"]]) {
    const dt = document.createElement("dt"), dd = document.createElement("dd");
    dt.textContent = key; dd.textContent = value; el("trial").append(dt, dd);
  }
  el("history").replaceChildren(...(data.operations.length ? data.operations : [{event:"No release changes yet", detail:""}]).map(op => {
    const li = document.createElement("li"); li.textContent = op.event + (op.detail ? ": " + op.detail : ""); return li;
  }));
  table("gates", data.offline_gates.map(g => [g.name, fmt(g.value), g.operator + " " + fmt(g.threshold), g.passed ? "Pass" : "Fail"]), 3);
  table("models", Object.entries(data.release_report.releases).map(([id, r]) => [id, fmt(r.test.f1), fmt(r.test.recall), fmt(r.cpu_p95_ms) + " ms"]));
  table("recent", data.recent.length ? data.recent.map(r => [r.robot, r.model, fmt(r.probability), r.alert ? "Fault alert" : "Normal", r.label == null ? "—" : r.label ? "Fault" : "Normal", fmt(r.inference_ms) + " ms"]) : [["Replay telemetry to populate this table.", "", "", "", "", ""]], 3);
  controls();
}
function notice(message, error = false) {
  el("notice").textContent = message;
  el("notice").classList.toggle("error", error);
}
async function run(path, message, body) {
  busy = true; controls(); notice("Working…");
  try {
    const data = await request(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: body == null ? undefined : JSON.stringify(body)});
    await refresh();
    notice(typeof message === "function" ? message(data) : message);
  } catch (err) { notice(err.message, true); }
  finally { busy = false; controls(); }
}
controls();
el("canary").onclick = () => run("/api/releases/canary", "Canary started. Replay labeled windows to collect evidence.", {candidate: state.active === "v1" ? "v2" : "v1", percent: 40});
el("replay").onclick = () => run("/api/replay", d => "Replayed " + d.windows + " windows, including " + d.canary + " routed to the canary.");
el("promote").onclick = () => run("/api/releases/promote", d => "Promoted " + d.active + ".");
el("rollback").onclick = () => run("/api/releases/rollback", d => "All traffic is now routed to " + d.active + ".");
refresh().then(() => notice("Ready. Start a canary or replay telemetry.")).catch(err => notice(err.message, true));
setInterval(() => { if (!busy) refresh().catch(() => {}); }, 5000);

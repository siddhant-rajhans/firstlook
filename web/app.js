// firstlook browser playground — runs the real firstlook in Pyodide (WASM).
// WHEEL must match the version built into web/ (the pages workflow builds it).
const WHEEL = "firstlook-0.5.0-py3-none-any.whl";

const $ = (id) => document.getElementById(id);
const status = (m) => { $("status").textContent = m; };

let pyodide, ready = false, csvText = null, sklearnLoaded = false;

async function boot() {
  try {
    pyodide = await loadPyodide();
    status("loading data libraries…");
    await pyodide.loadPackage(["micropip", "numpy", "pandas"]);
    const micropip = pyodide.pyimport("micropip");
    const wheelUrl = new URL(WHEEL, location.href).href;
    status("installing firstlook…");
    await micropip.install([wheelUrl, "plotly"]);
    pyodide.runPython(await (await fetch("driver.py")).text());
    ready = true;
    status("ready — drop a CSV.");
  } catch (e) {
    status("failed to start: " + e);
  }
}

function loadCSV(text) {
  csvText = text;
  if (!ready) { status("still booting — try again in a moment."); return; }
  try {
    pyodide.globals.set("CSV", csvText);
    const cols = JSON.parse(pyodide.runPython("columns(CSV)"));
    const sel = $("target");
    sel.innerHTML = '<option value="">(no target — explore / cluster)</option>' +
      cols.map((c) => `<option value="${c}">${c}</option>`).join("");
    if (cols.length) sel.value = cols[cols.length - 1];  // default: last column
    $("ctrls").style.display = "flex";
    status("pick a target, then Analyze.");
  } catch (e) {
    status("couldn't read that CSV: " + e);
  }
}

async function analyze() {
  if (!ready || csvText == null) return;
  const target = $("target").value || null;
  const fit = $("fit").checked;
  $("go").disabled = true;
  try {
    if (fit && !sklearnLoaded) {
      status("loading scikit-learn (one-time)…");
      await pyodide.loadPackage("scikit-learn");
      sklearnLoaded = true;
    }
    status("analyzing…");
    pyodide.globals.set("CSV", csvText);
    pyodide.globals.set("TGT", target);
    pyodide.globals.set("FIT", fit);
    const out = JSON.parse(pyodide.runPython("run(CSV, TGT, FIT, True)"));
    $("card").innerHTML = out.card;
    const fig = JSON.parse(out.fig);
    Plotly.newPlot("plot", fig.data, fig.layout, { displayModeBar: false, responsive: true });
    status("done.");
  } catch (e) {
    status("error: " + e);
  } finally {
    $("go").disabled = false;
  }
}

$("drop").addEventListener("click", () => $("file").click());
$("file").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) f.text().then(loadCSV); });
$("drop").addEventListener("dragover", (e) => { e.preventDefault(); $("drop").classList.add("over"); });
$("drop").addEventListener("dragleave", () => $("drop").classList.remove("over"));
$("drop").addEventListener("drop", (e) => {
  e.preventDefault(); $("drop").classList.remove("over");
  const f = e.dataTransfer.files[0]; if (f) f.text().then(loadCSV);
});
$("sample").addEventListener("click", (e) => {
  e.preventDefault();
  fetch("sample_iris.csv").then((r) => r.text()).then(loadCSV);
});
$("go").addEventListener("click", analyze);

boot();

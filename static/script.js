document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("upload-form");
  const resultContainer = document.getElementById("result-container");
  const loader = document.getElementById("loader");
  const processingMsg = document.getElementById("processing-msg");
  const similarityProgress = document.getElementById("similarity-progress");
  const filenameDisplay = document.getElementById("filename-display");
  const downloadBtn = document.getElementById("download-btn");
  const logList = document.getElementById("log-list");

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    const formData = new FormData(form);
    loader.style.display = "block";
    processingMsg.style.display = "block";
    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.text())
      .then((html) => {
        resultContainer.innerHTML = html;
        loader.style.display = "none";
        processingMsg.style.display = "none";
        const fileInput = form.querySelector("input[type='file']");
        const fileName = fileInput.files[0].name;
        filenameDisplay.textContent = `Filename: ${fileName}`;
        filenameDisplay.setAttribute("data-filename", fileName);
        downloadBtn.style.display = "inline-block";
        addLog("File uploaded successfully.");
      })
      .catch((err) => {
        alert("Upload failed.");
        loader.style.display = "none";
        processingMsg.style.display = "none";
      });
  });

  window.performOp = function (type) {
    const filename = filenameDisplay.getAttribute("data-filename");
    if (!filename) return alert("Upload a file first.");
    const formData = new FormData();
    formData.append("filename", filename);
    loader.style.display = "block";
    processingMsg.style.display = "block";
    fetch(`/operation/${type}`, {
      method: "POST",
      body: formData,
    })
      .then((res) => res.text())
      .then((html) => {
        resultContainer.innerHTML = html;
        loader.style.display = "none";
        processingMsg.style.display = "none";
        addLog(`Operation ${type} performed.`);
      });
  };

  window.analyze = function (type) {
    const filename = filenameDisplay.getAttribute("data-filename");
    if (!filename) return alert("Upload a file first.");
    const formData = new FormData();
    formData.append("filename", filename);
    loader.style.display = "block";
    processingMsg.style.display = "block";
    fetch(`/analyze/${type}`, {
      method: "POST",
      body: formData,
    })
      .then((res) => res.json())
      .then((data) => {
        const chartArea = document.getElementById("chart-area");
        chartArea.innerHTML = `<img src="${data.chart}" class="analysis-graph">`;
        const downloadGraph = document.getElementById("download-graph-link");
        downloadGraph.href = data.chart;
        downloadGraph.style.display = "inline-block";
        loader.style.display = "none";
        processingMsg.style.display = "none";
        addLog(`Analysis ${type} done.`);
      });
  };

  window.calculateSimilarity = function () {
    const filename = filenameDisplay.getAttribute("data-filename");
    if (!filename) return alert("Upload a file first.");
    const formData = new FormData();
    formData.append("filename", filename);
    loader.style.display = "block";
    processingMsg.style.display = "block";
    similarityProgress.innerText = "Starting...";

    const interval = setInterval(() => {
      fetch(`/progress/${filename}`)
        .then((res) => res.json())
        .then((data) => {
          similarityProgress.innerText = `Progress: ${data.progress}`;
        });
    }, 500);

    fetch("/similarity", {
      method: "POST",
      body: formData,
    })
      .then((res) => res.text())
      .then((html) => {
        clearInterval(interval);
        similarityProgress.innerText = "Similarity analysis complete.";
        resultContainer.innerHTML = html;
        loader.style.display = "none";
        processingMsg.style.display = "none";
        addLog("Similarity analysis completed.");
      });
  };

  function addLog(msg) {
    const li = document.createElement("li");
    li.textContent = `${new Date().toLocaleTimeString()}: ${msg}`;
    logList.appendChild(li);
  }
});

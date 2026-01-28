const fileNameEl = document.getElementById('fileName');
const fileInput = document.getElementById('audioFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const dropZone = document.getElementById('dropZone');
const resultCard = document.getElementById('resultCard');
const riskLabel = document.getElementById('riskLabel');
const confidenceEl = document.getElementById('confidence');
const disclaimerText = document.getElementById('disclaimerText');

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    fileNameEl.innerText = "Selected file: " + fileInput.files[0].name;
  } else {
    fileNameEl.innerText = "";
  }
});

// drag/drop visual feedback
['dragenter','dragover'].forEach(evt => {
  dropZone.addEventListener(evt, (e)=>{e.preventDefault(); dropZone.classList.add('drag');});
});
['dragleave','drop'].forEach(evt => {
  dropZone.addEventListener(evt, (e)=>{e.preventDefault(); dropZone.classList.remove('drag');});
});

// when file is dropped, set file input
dropZone.addEventListener('drop', (e)=>{
  e.preventDefault();
  const dt = e.dataTransfer;
  if(dt && dt.files && dt.files.length) fileInput.files = dt.files;
});

analyzeBtn.addEventListener('click', async () => {
  if (!fileInput.files || fileInput.files.length === 0) {
    alert('Please select an audio file first.');
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.innerText = 'Analyzing...';

  const form = new FormData();
  form.append('file', fileInput.files[0]);

  try {
    const res = await fetch('/screen', { method: 'POST', body: form });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Server error');
    }

    // Set UI
    resultCard.classList.remove('hidden');
    confidenceEl.innerText = `${(data.confidence * 100).toFixed(2)} %`;
    disclaimerText.innerText = data.disclaimer || '';

    // Handle two-class output
    const level = (data.risk_level || '').toLowerCase();
    riskLabel.className = 'value badge';

    if (level === 'low' || level === 'negative') {
      riskLabel.innerText = 'Low';
      riskLabel.classList.add('low');
    } else if (level === 'high' || level === 'positive') {
      riskLabel.innerText = 'High';
      riskLabel.classList.add('high');
    } else {
      riskLabel.innerText = '-';
    }

  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.innerText = 'Analyze';
  }
});

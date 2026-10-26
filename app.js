const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
let drawing = false;
let sampleId = 0;

function clearCanvas() { ctx.fillStyle = '#000'; ctx.fillRect(0,0,canvas.width,canvas.height); }
clearCanvas();
ctx.lineWidth = 24; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#fff';
canvas.addEventListener('pointerdown', e => { drawing=true; ctx.beginPath(); ctx.moveTo(e.offsetX,e.offsetY); });
canvas.addEventListener('pointermove', e => { if(!drawing) return; ctx.lineTo(e.offsetX,e.offsetY); ctx.stroke(); });
window.addEventListener('pointerup', () => drawing=false);

function renderBars(values) {
  document.getElementById('bars').innerHTML = values.map((v,i)=>`
    <div class="bar-row"><b>${i}</b><div class="bar-track"><div class="bar-fill" style="width:${(v*100).toFixed(2)}%"></div></div><span>${(v*100).toFixed(2)}%</span></div>`).join('');
}
renderBars(Array(10).fill(0));

async function loadSample() {
  const data = await (await fetch(`/api/sample?id=${sampleId++}`)).json();
  const image = new Image(); image.onload = () => { clearCanvas(); ctx.drawImage(image,0,0,280,280); }; image.src = data.image;
  document.getElementById('sampleInfo').textContent = `MNIST 测试样本 ${data.id}　真实标签 ${data.label}`;
  document.getElementById('prediction').textContent = '-';
  document.getElementById('confidence').textContent = '-';
  renderBars(Array(10).fill(0));
}

async function predict() {
  document.getElementById('status').textContent = '正在推理';
  const response = await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:canvas.toDataURL('image/png')})});
  const data = await response.json();
  document.getElementById('prediction').textContent = data.prediction;
  document.getElementById('confidence').textContent = `${(data.confidence*100).toFixed(2)}%`;
  document.getElementById('status').textContent = '识别完成';
  renderBars(data.probabilities);
}

document.getElementById('clearBtn').onclick = () => { clearCanvas(); document.getElementById('sampleInfo').textContent='可在画布中手写数字'; };
document.getElementById('sampleBtn').onclick = loadSample;
document.getElementById('predictBtn').onclick = predict;

document.getElementById('demoTab').onclick = () => {
  document.getElementById('demoPanel').classList.remove('hidden'); document.getElementById('resultsPanel').classList.add('hidden');
  document.getElementById('demoTab').classList.add('nav-active'); document.getElementById('resultsTab').classList.remove('nav-active');
};
document.getElementById('resultsTab').onclick = async () => {
  document.getElementById('demoPanel').classList.add('hidden'); document.getElementById('resultsPanel').classList.remove('hidden');
  document.getElementById('resultsTab').classList.add('nav-active'); document.getElementById('demoTab').classList.remove('nav-active');
  const m = await (await fetch('/api/metrics')).json();
  document.getElementById('metrics').innerHTML = [
    ['测试准确率',`${(m.test_accuracy*100).toFixed(2)}%`],['错误样本',`${m.test_error_count} / ${m.test_size}`],['批量推理速度',`${m.inference_fps.toFixed(0)} FPS`],['模型参数量',m.parameter_count.toLocaleString()]
  ].map(([a,b])=>`<div class="metric"><span>${a}</span><strong>${b}</strong></div>`).join('');
};
loadSample();

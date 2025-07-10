const testCases = [
  {
    name: 'Simple greeting',
    input: 'Hello',
    expected: (output) => typeof output === 'string' && output.toLowerCase().includes('hello'),
    notes: 'Should respond appropriately to a simple greeting.'
  },
  {
    name: 'Math question',
    input: 'What is 2 + 2?',
    expected: (output) => /4/.test(output),
    notes: 'Should answer a basic math question.'
  },
  {
    name: 'Edge case: empty input',
    input: '',
    expected: (output) => output && output.length > 0,
    notes: 'Should handle empty input gracefully.'
  },
  {
    name: 'Malformed input',
    input: '{ this is not valid JSON',
    expected: (output) => typeof output === 'string',
    notes: 'Should not crash on malformed input.'
  },
  {
    name: 'Long input',
    input: 'a'.repeat(1000),
    expected: (output) => typeof output === 'string',
    notes: 'Should handle long input.'
  }
];

document.getElementById('runDiagnostics').addEventListener('click', async () => {
  const endpoint = document.getElementById('endpoint').value.trim();
  const reportDiv = document.getElementById('report');
  reportDiv.textContent = 'Running diagnostics...';

  if (!endpoint) {
    reportDiv.textContent = 'Please enter an endpoint URL.';
    return;
  }

  let results = [];
  for (const test of testCases) {
    let output = '';
    let status = 'FAIL';
    let error = null;
    let responseTime = null;
    try {
      const start = Date.now();
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: test.input })
      });
      responseTime = Date.now() - start;
      const data = await response.json();
      output = data.output || JSON.stringify(data);
      if (test.expected(output)) {
        status = 'PASS';
      } else {
        status = 'FAIL';
      }
    } catch (err) {
      error = err.toString();
      status = 'ERROR';
    }
    results.push({
      name: test.name,
      status,
      notes: test.notes,
      output,
      error,
      responseTime
    });
  }

  // Generate report
  let passCount = results.filter(r => r.status === 'PASS').length;
  let failCount = results.filter(r => r.status === 'FAIL').length;
  let errorCount = results.filter(r => r.status === 'ERROR').length;
  let overall = '🟢 Healthy';
  if (failCount > 0) overall = '🟡 Needs Attention';
  if (errorCount > 0) overall = '🔴 Critical Issues';

  let html = `<b>Diagnostics Report</b><br><b>Overall:</b> ${overall}<br><br>`;
  html += '<table style="width:100%;font-size:0.95em;border-collapse:collapse;">';
  html += '<tr><th style="border-bottom:1px solid #ccc;">Test</th><th style="border-bottom:1px solid #ccc;">Status</th><th style="border-bottom:1px solid #ccc;">Response Time (ms)</th><th style="border-bottom:1px solid #ccc;">Notes</th></tr>';
  for (const r of results) {
    html += `<tr><td>${r.name}</td><td>${r.status}</td><td>${r.responseTime !== null ? r.responseTime : '-'}</td><td>${r.notes}${r.error ? '<br><span style=\'color:red\'>' + r.error + '</span>' : ''}</td></tr>`;
  }
  html += '</table>';
  reportDiv.innerHTML = html;
}); 
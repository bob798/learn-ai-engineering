/* ============================================
   AI Programming — Shared Interactions
   ============================================ */

// === Tab Switch ===
function switchTab(i) {
  var tabs = document.querySelectorAll('.nav-tab');
  var cards = document.querySelectorAll('.card');
  tabs.forEach(function(t, j) { t.classList.toggle('active', j === i); });
  cards.forEach(function(c, j) { c.classList.toggle('active', j === i); });
  setTimeout(function() { cards[i].scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 50);
}

// === Code Collapse ===
function toggleCode(id) {
  document.getElementById(id).classList.toggle('open');
}

// === Keyword Drawer ===
function openKw(k) {
  var data = window.KW_DATA;
  if (!data) return;
  var d = data[k];
  if (!d) return;
  document.getElementById('dIc').innerHTML = d.icon;
  document.getElementById('dTi').textContent = d.title;
  document.getElementById('dEn').textContent = d.en;
  var h = '';
  d.s.forEach(function(s) {
    var lc = s.t === 'i' ? 'ds-i' : s.t === 't' ? 'ds-t' : 'ds-b';
    var bc = s.b === 'orange' ? 'dsb-orange' : s.b === 'purple' ? 'dsb-purple' : 'dsb-blue';
    h += '<div class="ds"><div class="ds-l ' + lc + '">' + s.l + '</div><div class="ds-box ' + bc + '">' + s.c + '</div></div>';
  });
  document.getElementById('dBd').innerHTML = h;
  document.getElementById('dov').classList.add('show');
  document.getElementById('drw').classList.add('show');
}

function closeKw() {
  document.getElementById('dov').classList.remove('show');
  document.getElementById('drw').classList.remove('show');
}

// === Init: mark cards ready after load ===
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.card').forEach(function(c) { c.classList.add('ready'); });
});

// ────────────────────────────────────────────────────────────
//  static/script.js — полноразмерный, мульти-пользовательский
// ────────────────────────────────────────────────────────────

// ───────── Telegram WebApp API + fallback для обычного браузера
if (!window.Telegram || !Telegram.WebApp) {
  console.warn('⚠ Telegram.WebApp не найден — локальный режим.');
  window.Telegram = {
    WebApp: {
      sendData: d => console.log('tg.sendData →', d),
      close:    () => console.log('tg.close()'),
      initDataUnsafe: {}
    }
  };
}
const tg = Telegram.WebApp;

/* -----------------------------------------------------------
   Авто-сброс, если Web-App запущен с  ?reset=1  или
   если start_param (deep-link) == 'reset'
----------------------------------------------------------- */
(function () {
  const urlParam   = new URLSearchParams(location.search).get('reset');
  const startParam = tg.initDataUnsafe?.start_param;          // deep-link
  const needReset  = urlParam === '1' || startParam === 'reset';

  // чтобы не уйти в бесконечный цикл, проверяем флаг
  if (needReset && !sessionStorage.getItem('didHardReset')) {
    sessionStorage.setItem('didHardReset', 'yes');  // один раз за сессию
    // чистим все наши ключи и мгновенно перезагружаем
    localStorage.removeItem('tgUser');
    localStorage.removeItem('allSettings');
    location.href = location.origin + location.pathname;     // перезапуск без ?reset
  }
})();


// ───────── Удобный доступ к текущему userId
const getUserId = () => {
  const ls = JSON.parse(localStorage.getItem('tgUser') || 'null');
  return tg.initDataUnsafe?.user?.id || ls?.id || null;
};

/* ────────────────────────────────────────────
   Скрываем оверлей, ЕСЛИ запись tgUser уже есть
──────────────────────────────────────────── */
if (localStorage.getItem('tgUser')) {
  const hide = () => {
    const lo = document.getElementById('loginOverlay');
    const ao = document.getElementById('authAnimationOverlay');
    if (lo) lo.style.display = 'none';
    if (ao) ao.style.display = 'none';
  };
  if (document.readyState !== 'loading') hide();
  else window.addEventListener('DOMContentLoaded', hide);
}

/* ===========================================================
   0) localStorage helpers  (словарь {id:settings})
   =========================================================== */
const DEFAULT_SETTINGS = {
  temperature: 0.7, top_k: 50, top_p: 0.85,
  repetition_penalty: 2.0, length_penalty: 1.0, speed: 1.0,
  notifications: true, dotX: 50, dotY: 50
};
const STORAGE_KEY = 'allSettings';
function getStoredSettings(){
  const db = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
  const u  = getUserId() || 'anon';
  return db[u] ? { ...DEFAULT_SETTINGS, ...db[u] } : { ...DEFAULT_SETTINGS };
}
function saveSettings(s){
  const db = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
  const u  = getUserId() || 'anon';
  db[u] = s;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(db));
}

/* ===========================================================
   1)  Шаблоны (5 вершин)
   =========================================================== */
const TEMPLATES = [
  { name:'Креатив', settings:{ temperature:1, top_k:100, top_p:1,   repetition_penalty:1, length_penalty:0.5, speed:1 } },
  { name:'Строго',  settings:{ temperature:0, top_k:0,   top_p:0,   repetition_penalty:3, length_penalty:2,   speed:1 } },
  { name:'Быстро',  settings:{ temperature:0.7, top_k:50, top_p:0.85, repetition_penalty:2, length_penalty:1, speed:2 } },
  { name:'Медленно',settings:{ temperature:0.7, top_k:50, top_p:0.85, repetition_penalty:2, length_penalty:1, speed:0.1 } },
  { name:'Повтор',  settings:{ temperature:0.7, top_k:50, top_p:0.85, repetition_penalty:3, length_penalty:1, speed:1 } },
];

/* ===========================================================
   2)  DOM ready → авторизация + синхронизация
   =========================================================== */
document.addEventListener('DOMContentLoaded', () => {
  const tgId = tg.initDataUnsafe?.user?.id;
  if (tgId && !localStorage.getItem('tgUser')) {
    localStorage.setItem('tgUser', JSON.stringify({ id: tgId }));
  }

  const userId = getUserId();
  const startUI = () => {
    initializeUI();
    if (userId) {
      document.getElementById('loginOverlay').style.display = 'none';
      document.getElementById('authAnimationOverlay').style.display = 'none';
    }
  };

  if (!userId) return startUI();

  fetch(`/get_user_settings?userId=${userId}`)
    .then(r => r.ok ? r.json() : null)
    .then(j => { if (j?.status === 'success') saveSettings(j.settings); })
    .catch(console.warn)
    .finally(startUI);
});

/* ===========================================================
   3)  Telegram Login Widget callback
   =========================================================== */
function onTelegramAuth(user) {
  localStorage.setItem('tgUser', JSON.stringify(user));
  document.getElementById('loginOverlay').style.display = 'none';
  const anim = document.getElementById('authAnimationOverlay');
  anim.style.display = 'flex';

  fetch('/telegram_auth', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(user)
  }).catch(console.error);

  setTimeout(() => { anim.style.display = 'none'; initializeUI(); }, 3000);
}

/* ===========================================================
   4)  UI инициализация
   =========================================================== */
function initializeUI() {
  injectVertexLabels();
  injectApplyButton();

  const s = getStoredSettings();

  // ── СНАЧАЛА позиционируем шарик (без interpolate) ──
  setDotPosition(s.dotX, s.dotY, /* init = */ true);

  // ── потом заполняем слайдеры / inputs сохранёнными значениями ──
  updateUIFromSettings(getStoredSettings());

  initOptions();
  highlightTab(0); centerTabInView(0);
}

/* -----------------------------------------------------------
   4.1)  подписи к вершинам
----------------------------------------------------------- */
function injectVertexLabels() {
  const wr=document.querySelector('.pentagon-wrapper');
  if(!wr||wr.querySelector('.vertex-label')) return;
  wr.style.position='relative';
  const POS=[{l:'50%',t:'-6px',tx:'-50%',ty:'-100%'},{l:'calc(80% + 12px)',t:'35%',tx:'0',ty:'-30%'},
             {l:'calc(68% + 12px)',t:'105%',tx:'0',ty:'-50%'},{l:'20%',t:'calc(96% + 12px)',tx:'-50%',ty:'0'},
             {l:'54px',t:'36%',tx:'-100%',ty:'-50%'}];
  TEMPLATES.forEach((tpl,i)=>{
    const d=document.createElement('div');d.className='vertex-label';d.textContent=tpl.name;
    Object.assign(d.style,{position:'absolute',left:POS[i].l,top:POS[i].t,
      transform:`translate(${POS[i].tx},${POS[i].ty})`,color:'#3cd4cb',fontSize:'12px',whiteSpace:'nowrap',pointerEvents:'none'});
    wr.appendChild(d);
  });
}

/* -----------------------------------------------------------
   4.2)  кнопка «Применить»
----------------------------------------------------------- */
function injectApplyButton() {
  if(document.getElementById('applyBtn')) return;
  const blk=document.getElementById('settingsBlock'); if(!blk) return;
  const b=document.createElement('button');b.id='applyBtn';b.textContent='Применить настройки';
  Object.assign(b.style,{display:'block',margin:'0 auto 16px',padding:'10px 24px',
    background:'#3cd4cb',color:'#000',border:'none',borderRadius:'8px',fontWeight:'bold',cursor:'pointer'});
  b.onclick=applyAndSave; blk.parentNode.insertBefore(b,blk);
}

/* -----------------------------------------------------------
   4.3)  переключатель уведомлений
----------------------------------------------------------- */
function initOptions() {
  const t=document.getElementById('notificationToggle');
  if(t) t.addEventListener('change',()=>sync(t));
}

/* ===========================================================
   5)  apply + save
   =========================================================== */
function applyAndSave() {
  const userId = getUserId();
  if (!userId) {
    alert('Сначала войдите через Telegram, чтобы сохранить настройки.');
    return;
  }

  const s = getStoredSettings();
  document.querySelectorAll('input[data-key]').forEach(el=>{
    const k=el.dataset.key; s[k]= el.type==='checkbox'? el.checked : parseFloat(el.value);
  });
  saveSettings(s);

  if (tg?.sendData && tg.initDataUnsafe?.user?.id) {
    tg.sendData(JSON.stringify({action:'save_settings',settings:s}));
  }

  fetch('/save_user_settings', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({userId,settings:s})
  }).then(r=>r.json()).then(console.log).catch(console.error);
}

/* --- отложенный sendData ------------------------------- */
let sendSettingsTimeout=null;
function scheduleSettingsSend(){
  clearTimeout(sendSettingsTimeout);
  sendSettingsTimeout=setTimeout(()=>{
    if(tg?.sendData && tg.initDataUnsafe?.user?.id)
      tg.sendData(JSON.stringify({action:'save_settings',settings:getStoredSettings()}));
  },1000);
}

/* ===========================================================
   6)  пятиугольник + точка (drag / click)
   =========================================================== */
function pointInPentagon(px,py,rect,extra=0){
  const c=document.createElement('canvas');c.width=rect.width;c.height=rect.height;
  const ctx=c.getContext('2d'),cx=rect.width/2,cy=rect.height/2,r=rect.width/2+extra;
  ctx.beginPath();
  for(let i=0;i<5;i++){const a=Math.PI/2+i*2*Math.PI/5,x=cx+r*Math.cos(a),y=cy-r*Math.sin(a);
    i?ctx.lineTo(x,y):ctx.moveTo(x,y);} ctx.closePath();
  return ctx.isPointInPath(px-rect.left,py-rect.top);
}

const pent=document.getElementById('pentagon'), dot=document.getElementById('dot');
let dragging=false;
const VERT=[{x:50,y:0},{x:90,y:27},{x:77,y:90},{x:23,y:90},{x:10,y:27}];

/* ───── изменено: добавлен аргумент init (по-умолчанию false) ───── */
function setDotPosition(px,py,init=false){
  const r=pent.getBoundingClientRect(),dotR=dot.offsetWidth/2;
  const ax=r.left+r.width*px/100,ay=r.top+r.height*py/100;
  if(!pointInPentagon(ax,ay,r,dotR)) return;
  dot.style.left=`${px}%`;dot.style.top=`${py}%`;

  const s=getStoredSettings();
  s.dotX = px;
  s.dotY = py;

  if (!init) {                 // ← обычный drag / click
    saveSettings(s);
    interpolate(px, py);
  }
}

function startDrag(e){dragging=true;moveDot(e);}
function moveDot(e){
  if(!dragging)return;
  const r=pent.getBoundingClientRect();
  const x=e.touches?e.touches[0].clientX:e.clientX;
  const y=e.touches?e.touches[0].clientY:e.clientY;
  setDotPosition((x-r.left)/r.width*100,(y-r.top)/r.height*100);
}
dot.addEventListener('mousedown',startDrag);dot.addEventListener('touchstart',startDrag,{passive:true});
pent.addEventListener('mousedown',startDrag);pent.addEventListener('touchstart',startDrag,{passive:true});
document.addEventListener('mouseup',()=>dragging=false);document.addEventListener('touchend',()=>dragging=false);
document.addEventListener('mousemove',moveDot);document.addEventListener('touchmove',moveDot,{passive:false});

function interpolate(px,py){
  const w=[],EPS=1e-4;let sum=0;
  VERT.forEach(v=>{const d=1/(Math.hypot(px-v.x,py-v.y)+EPS);w.push(d);sum+=d;});
  w.forEach((_,i)=>w[i]/=sum);
  const s=getStoredSettings();let changed=false;
  Object.keys(TEMPLATES[0].settings).forEach(k=>{
    let v=0;for(let i=0;i<5;i++) v+=w[i]*TEMPLATES[i].settings[k];
    if(k==='top_k') v=Math.round(v); v=+v.toFixed(3);
    if(Math.abs(s[k]-v)>1e-3){s[k]=v;changed=true;}
  });
  if(changed){saveSettings(s);updateUIFromSettings(s);scheduleSettingsSend();}
}

/* ===========================================================
   7)  Tabs + scroll
   =========================================================== */
let curPage=0;
function switchTab(i){const p=document.querySelectorAll('.page');if(i<0||i>=p.length)return;
  p[i].scrollIntoView({behavior:'smooth'});curPage=i;highlightTab(i);centerTab(i);}
function highlightTab(i){document.querySelectorAll('.tab').forEach((t,idx)=>t.classList.toggle('active',idx===i));}
function centerTab(i){
  const tb=document.querySelectorAll('#topTabs .tab')[i]; if(!tb)return;
  const tr=tb.getBoundingClientRect(),wr=document.getElementById('topTabs').getBoundingClientRect();
  document.getElementById('topTabs').scrollBy({left:(tr.left+tr.width/2)-(wr.left+wr.width/2),behavior:'smooth'});}
document.getElementById('scrollContainer').addEventListener('scroll',()=>{
  const i=Math.round(document.getElementById('scrollContainer').scrollLeft/window.innerWidth);
  if(i!==curPage){curPage=i;highlightTab(i);centerTab(i);}});

/* ===========================================================
   8)  UI ↔ settings
   =========================================================== */
function updateUIFromSettings(s){
  document.querySelectorAll('input[type="range"],input[type="number"]').forEach(el=>{
    const k=el.dataset.key; if(k in s && typeof s[k]!=='boolean') el.value=s[k];});
  const t=document.getElementById('notificationToggle'); if(t) t.checked=s.notifications;
}
function resetAllSettings(){
  saveSettings({...DEFAULT_SETTINGS});
  updateUIFromSettings(DEFAULT_SETTINGS); setDotPosition(50,50); scheduleSettingsSend();
}
function sync(el){
  const k=el.dataset.key, pair=el.type==='range'?el.nextElementSibling:el.previousElementSibling;
  if(pair) pair.value=el.type==='checkbox'?el.checked:el.value;
  const s=getStoredSettings(); s[k]=el.type==='checkbox'?el.checked:parseFloat(el.value);
  saveSettings(s); scheduleSettingsSend();
}
document.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('input[type="range"]').forEach(r=>{
    r.addEventListener('dblclick',()=>{
      const k=r.dataset.key,s=getStoredSettings(); s[k]=DEFAULT_SETTINGS[k];
      saveSettings(s); r.value=s[k]; const n=r.nextElementSibling; if(n&&n.type==='number') n.value=s[k];
      scheduleSettingsSend();});});
});

/* ===========================================================
   9)  Подсказки, тарифы, опции
   =========================================================== */
function showTooltip(t){document.getElementById('tooltipText').innerText=t;
  document.getElementById('tooltipModal').classList.add('show');}
function closeTooltip(){document.getElementById('tooltipModal').classList.remove('show');}
function selectTariff(e,el,n){e.stopPropagation();if(el.classList.contains('selected'))return;
  document.querySelectorAll('.tariff-btn').forEach(b=>{b.classList.remove('selected');
    const d=b.querySelector('.tariff-details');if(d)b.removeChild(d);});
  el.classList.add('selected');
  const perks={'Base Free':['Обычный доступ','Ограниченные функции'],
               'Base+':['Все из Base Free','Поддержка','Больше настроек'],
               'Vip':['Все из Base+','Ранний доступ','Персонализация'],
               'Premium':['Все из Vip','Безлимит','Поддержка 24/7']};
  if(!perks[n]) return;
  const box=document.createElement('div'); box.className='tariff-details';
  perks[n].forEach(txt=>{const li=document.createElement('li');li.textContent=txt;box.appendChild(li);});
  const btn=document.createElement('button');btn.className='btn-confirm';btn.textContent='Оформление';
  box.appendChild(btn); el.appendChild(box);}
function handleTariffBackgroundClick(e){if(!e.target.classList.contains('tariff-btn'))
  document.querySelectorAll('.tariff-btn').forEach(b=>{b.classList.remove('selected');
    const d=b.querySelector('.tariff-details');if(d)b.removeChild(d);});}
function toggleSettings(){document.getElementById('settingsBlock').classList.toggle('open');
  document.getElementById('chevron').classList.toggle('open');}

/* ---------- ОБНОВЛЁННАЯ функция проверки аудио ---------- */
function checkAudio(){
  const f=document.getElementById('audioFile'),
        r=document.getElementById('audioCheckResult');

  if(!f.files||!f.files[0]){
    r.textContent='Файл не выбран.'; r.style.color='#ccc'; return;
  }

  const fd=new FormData();
  fd.append('audio', f.files[0]);

  r.textContent='⏳ Проверка...'; r.style.color='#ccc';

  fetch('/audio_check',{method:'POST',body:fd})
    .then(res=>res.json())
    .then(j=>{
      if(j.status==='ok'){
        r.textContent=j.result;
        r.style.color = j.result.includes('real') ? 'green' : 'red';
      }else{
        r.textContent='Ошибка: '+j.message;
        r.style.color='red';
      }
    })
    .catch(()=>{ r.textContent='Ошибка сети'; r.style.color='red'; });
}

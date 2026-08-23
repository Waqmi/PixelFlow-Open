const modalBackdrop = document.querySelector('#modal-backdrop');
const modalContent = document.querySelector('#modal-content');
const toast = document.querySelector('#toast');
let progressTimer = null;

const showToast = (message) => {
  toast.textContent = message;
  toast.classList.add('show');
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove('show'), 2200);
};

const closeModal = () => {
  modalBackdrop.classList.add('hidden');
  modalContent.replaceChildren();
};

const modalShell = (eyebrow, title, description, body, actions = '') => {
  modalContent.innerHTML = `
    <div class="modal-header"><span class="eyebrow">${eyebrow}</span><h2>${title}</h2><p>${description}</p></div>
    <div class="modal-body">${body}</div>${actions ? `<div class="settings-footer">${actions}</div>` : ''}`;
  modalBackdrop.classList.remove('hidden');
};

const openSettings = () => {
  const body = `
    <div class="settings-layout">
      <nav class="settings-nav"><h2>设置</h2>
        <button class="active" data-settings-tab="general">通用</button>
        <button data-settings-tab="brand">品牌</button>
        <button data-settings-tab="output">输出</button>
        <button data-settings-tab="advanced">高级</button>
      </nav>
      <div class="settings-content">
        <section class="setting-pane active" data-pane="general"><div class="settings-form">
          <div class="setting-group"><h4>默认输出位置</h4><select><option>跟随素材文件夹</option><option>上次使用位置</option><option>固定输出目录</option></select></div>
          <label class="check-option"><input type="checkbox" checked> 生成完成后自动打开输出文件夹</label>
          <div class="setting-inline"><label>主题<select><option>跟随系统</option><option>浅色</option><option>深色</option></select></label><label>语言<select><option>简体中文</option></select></label></div>
        </div></section>
        <section class="setting-pane" data-pane="brand"><div class="settings-form">
          <div class="logo-intro">品牌设置只管理长期默认值。四张 Logo 均需提供完整尺寸 PNG，App 会在保存前检查文件。</div>
          <div class="logo-set"><div class="logo-set-title"><b>方图 Logo</b><span>1440×1440</span></div><div class="logo-file-grid">${logoFile('方图深色 Logo', '浅色背景使用')} ${logoFile('方图浅色 Logo', '深色背景使用')}</div></div>
          <div class="logo-set"><div class="logo-set-title"><b>竖图 Logo</b><span>1440×1920</span></div><div class="logo-file-grid">${logoFile('竖图深色 Logo', '浅色背景使用')} ${logoFile('竖图浅色 Logo', '深色背景使用')}</div></div>
          <div class="logo-rules"><strong>Logo 文件规范</strong><ul><li>PNG、RGBA 透明背景、完整尺寸画布</li><li>不带白底、边框、阴影或额外文字</li><li>深色版用于浅色背景，浅色版用于深色背景</li></ul></div>
        </div></section>
        <section class="setting-pane" data-pane="output"><div class="settings-form"><div class="setting-group"><h4>默认输出规格</h4>${['1440×1440', '1440×1920', '1125×1500'].map(size => `<label class="check-option"><input type="checkbox" checked>${size}</label>`).join('')}</div><div class="setting-group"><h4>默认启用</h4>${['Logo', '自动识别素材类型', '自动生成模特主图'].map(label => `<label class="check-option"><input type="checkbox" checked>${label}</label>`).join('')}${['唯品专享图', '自动生成卖点图'].map(label => `<label class="check-option"><input type="checkbox">${label}</label>`).join('')}</div></div></section>
        <section class="setting-pane" data-pane="advanced"><div class="settings-form"><div class="logo-intro">高级设置先保留入口。品类比例、文件大小限制等当前任务参数仍在主界面的“高级设置”中调整。</div></div></section>
      </div>
    </div>`;
  modalShell('APP DEFAULTS', '设置', '一次设置，长期复用；主界面仍可临时覆盖。', body, '<button class="danger-link" data-reset-brand>恢复默认品牌设置</button><button class="primary-button" data-save-modal>保存设置</button>');
  modalContent.querySelectorAll('[data-settings-tab]').forEach(button => button.addEventListener('click', () => {
    modalContent.querySelectorAll('[data-settings-tab]').forEach(item => item.classList.remove('active'));
    modalContent.querySelectorAll('[data-pane]').forEach(item => item.classList.remove('active'));
    button.classList.add('active');
    modalContent.querySelector(`[data-pane="${button.dataset.settingsTab}"]`).classList.add('active');
  }));
  modalContent.querySelectorAll('[data-replace-logo]').forEach(button => button.addEventListener('click', () => showToast(`${button.dataset.replaceLogo}：模拟选择 PNG 文件`)));
  modalContent.querySelector('[data-reset-brand]').addEventListener('click', () => showToast('已恢复 PixelFlow 内置品牌资源'));
  modalContent.querySelector('[data-save-modal]').addEventListener('click', () => { closeModal(); showToast('设置已保存'); });
};

const logoFile = (name, hint) => `<div class="logo-file"><div class="logo-preview">PF</div><div><b>${name}</b><small>${hint}</small><button class="replace-link" data-replace-logo="${name}">更换并检查</button></div></div>`;

const reviewCard = (type, name, description, extra = '') => `<article class="review-card"><div class="material-preview ${type}">${type === 'model' ? '模特图' : type === 'detail' ? '细节图' : '商品白底图'}</div><div><span class="review-tag ${type === 'model' ? '' : 'warn'}">${type === 'model' ? '模特图 · 单人全身' : '低置信度 · 待确认'}</span><h4>${name}</h4><p>${description}</p>${extra}<div class="review-actions"><button class="primary" data-review-action="confirm">确认</button><button data-review-action="change">修改分类</button></div></div></article>`;

const openReview = () => {
  const body = `<div class="review-grid">${reviewCard('product', 'PRTL001_08.jpg', '浅色背景，主体边缘接近画面边界。')} ${reviewCard('model', 'MODEL_02.jpg', '动作图，人物手部接近边缘，系统建议确认。')} ${reviewCard('detail', 'PRTL001_14.jpg', '细节图可自动匹配卖点：透气。', '<div class="candidate-row"><button>匹配：透气</button><button>匹配：防磨</button></div>')}</div>`;
  modalShell('REVIEW QUEUE · 3', '确认待处理素材', '只处理系统无法可靠判断的少数素材。', body);
  modalContent.querySelectorAll('[data-review-action]').forEach(button => button.addEventListener('click', () => { button.closest('.review-card').remove(); showToast(button.dataset.reviewAction === 'confirm' ? '已确认，继续自动生成' : '已切换为手动分类'); }));
};

const openSellingPoints = () => {
  const points = ['透气', '防滑', '防磨', '支撑', '快干'];
  const body = `<div class="point-list">${points.map((point, index) => `<div class="point-row"><strong>卖点 ${index + 1}</strong><span>${point}</span><button class="trash" title="删除">×</button></div>`).join('')}</div><div class="batch-box"><h4>批量粘贴</h4><p>每行一个卖点，系统会自动去重并匹配细节图。</p><textarea placeholder="例如：\n透气\n防滑\n轻量"></textarea></div>`;
  modalShell('SELLING POINTS', '产品卖点', '先配置品牌卖点，生成时自动匹配细节图。', body, '<button class="outline-button" data-cancel-modal>取消</button><button class="primary-button" data-save-points>保存卖点</button>');
  modalContent.querySelectorAll('.trash').forEach(button => button.addEventListener('click', () => button.closest('.point-row').remove()));
  modalContent.querySelector('[data-cancel-modal]').addEventListener('click', closeModal);
  modalContent.querySelector('[data-save-points]').addEventListener('click', () => { closeModal(); showToast('卖点配置已更新'); });
};

document.querySelectorAll('[data-modal]').forEach(button => button.addEventListener('click', () => {
  if (button.dataset.modal === 'settings-modal') openSettings();
  if (button.dataset.modal === 'review-modal') openReview();
  if (button.dataset.modal === 'selling-points-modal') openSellingPoints();
}));
document.querySelector('#modal-close').addEventListener('click', closeModal);
modalBackdrop.addEventListener('click', event => { if (event.target === modalBackdrop) closeModal(); });
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeModal(); });

document.querySelectorAll('[data-toggle-card]').forEach(card => card.addEventListener('click', () => card.classList.toggle('active')));
document.querySelectorAll('[data-size-card]').forEach(card => card.addEventListener('click', () => card.classList.toggle('selected')));
document.querySelector('[data-advanced-trigger]').addEventListener('click', event => {
  event.currentTarget.classList.toggle('open');
  document.querySelector('#advanced-panel').classList.toggle('hidden');
});
document.querySelectorAll('.choose-button').forEach(button => button.addEventListener('click', () => {
  document.querySelector(`#${button.dataset.target}`).value = button.dataset.target === 'source-path' ? '示例素材 / PRSL005-2 红' : '示例输出 / 主图';
  showToast('已选择示例路径');
}));
document.querySelector('#start-button').addEventListener('click', event => {
  const button = event.currentTarget;
  const stop = document.querySelector('#stop-button');
  const wrap = document.querySelector('#progress-wrap');
  const bar = document.querySelector('#progress-bar');
  const label = document.querySelector('#progress-label');
  const title = document.querySelector('#status-title');
  const copy = document.querySelector('#status-copy');
  let value = 0;
  button.classList.add('hidden');
  stop.classList.remove('hidden');
  wrap.classList.remove('hidden');
  title.textContent = '正在识别素材';
  copy.textContent = '先自动分类，再把低置信度项目送入待确认。';
  progressTimer = window.setInterval(() => {
    value = Math.min(100, value + 10);
    bar.style.width = `${value}%`;
    label.textContent = `${value}%`;
    if (value === 100) {
      window.clearInterval(progressTimer);
      stop.classList.add('hidden');
      button.classList.remove('hidden');
      title.textContent = '所有任务完成';
      copy.textContent = '已模拟完成 25 张素材处理。';
      showToast('模拟生成完成');
    }
  }, 220);
});
document.querySelector('#stop-button').addEventListener('click', () => {
  window.clearInterval(progressTimer);
  document.querySelector('#stop-button').classList.add('hidden');
  document.querySelector('#start-button').classList.remove('hidden');
  document.querySelector('#status-title').textContent = '已停止';
  document.querySelector('#status-copy').textContent = '可重新开始自动识别。';
  showToast('已停止模拟生成');
});

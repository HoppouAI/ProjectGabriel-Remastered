const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

let defaults = {};
let existingConfig = null;
let selectedTemplate = 'template';

async function init() {
  try {
    const res = await fetch('/api/defaults');
    defaults = await res.json();
    applyDefaults();
  } catch (e) {}

  // Load existing config.yml if present and prefill
  try {
    const res = await fetch('/api/load-config');
    const data = await res.json();
    if (data.exists && data.config) {
      existingConfig = data.config;
      prefillFromExisting(existingConfig);
      $('#overwriteWarning').style.display = 'block';
    }
  } catch (e) {}

  try {
    const res = await fetch('/api/audio-devices');
    const devices = await res.json();
    populateAudioDevices(devices);
    // After populating device options, select existing values
    if (existingConfig && existingConfig.audio) {
      if (existingConfig.audio.input_device != null)
        $('#audio_input').value = existingConfig.audio.input_device;
      if (existingConfig.audio.output_device != null)
        $('#audio_output').value = existingConfig.audio.output_device;
    }
  } catch (e) {}

  // Remove the old check-config call (load-config handles it now)

  $('#feat_memory').addEventListener('change', () => {
    $('#memoryOptions').classList.toggle('visible', $('#feat_memory').checked);
  });
  if ($('#feat_memory').checked) $('#memoryOptions').classList.add('visible');

  $('#feat_vision').addEventListener('change', () => {
    $('#visionOptions').classList.toggle('visible', $('#feat_vision').checked);
  });

  $('#gemini_model').addEventListener('change', updateThinkingFields);
  updateThinkingFields();

  $('#idle_chatbox_enabled').addEventListener('change', () => {
    $('#idleChatboxOptions').classList.toggle('visible', $('#idle_chatbox_enabled').checked);
  });

  // TTS provider conditional fields
  $('#tts_provider').addEventListener('change', updateTtsOptions);
  updateTtsOptions();

  // If prompts.yml already exists, auto-select skip to avoid overwriting
  try {
    const res = await fetch('/api/check-prompts');
    const data = await res.json();
    if (data.exists) {
      selectTemplate('skip');
      showToast('Existing prompts.yml detected, prompt step set to Skip to avoid overwriting.', 'success');
    }
  } catch (e) {}

  ['char_name','char_age','char_desc','char_location','char_background','char_personality'].forEach(id => {
    document.getElementById(id).addEventListener('input', updatePreview);
  });

  updateTemplateVisibility();
  updatePreview();

  // Clear field errors on input
  document.querySelectorAll('input, textarea, select').forEach(el => {
    el.addEventListener('input', () => {
      el.classList.remove('error');
      const errNote = el.parentElement.querySelector('.field-error');
      if (errNote) errNote.remove();
    });
  });
}

function applyDefaults() {
  if (defaults.app_name) $('#app_name').value = defaults.app_name;
  if (defaults.gemini) {
    if (defaults.gemini.model) {
      const opt = $(`#gemini_model option[value="${defaults.gemini.model}"]`);
      if (opt) opt.selected = true;
    }
    if (defaults.gemini.voice) {
      const opt = $(`#gemini_voice option[value="${defaults.gemini.voice}"]`);
      if (opt) opt.selected = true;
    }
    if (defaults.gemini.temperature != null) {
      $('#gemini_temp').value = defaults.gemini.temperature;
      $('#tempVal').textContent = defaults.gemini.temperature;
    }
    if (defaults.gemini.vad) {
      const vad = defaults.gemini.vad;
      if (vad.mode) { $('#vad_mode').value = vad.mode; updateVadOptions(); }
      if (vad.start_of_speech_sensitivity) $('#vad_start').value = vad.start_of_speech_sensitivity;
      if (vad.end_of_speech_sensitivity) $('#vad_end').value = vad.end_of_speech_sensitivity;
      if (vad.silence_duration_ms != null) {
        $('#vad_silence').value = vad.silence_duration_ms;
        $('#silenceVal').textContent = vad.silence_duration_ms + 'ms';
      }
    }
    if (defaults.gemini.thinking) {
      const t = defaults.gemini.thinking;
      if (t.level) $('#thinking_level').value = t.level;
      if (t.budget != null) $('#thinking_budget').value = t.budget;
    }
    updateThinkingFields();
  }
  if (defaults.vrchat) {
    if (defaults.vrchat.osc_ip) $('#osc_ip').value = defaults.vrchat.osc_ip;
    if (defaults.vrchat.osc_send_port) $('#osc_send_port').value = defaults.vrchat.osc_send_port;
    if (defaults.vrchat.osc_receive_port) $('#osc_recv_port').value = defaults.vrchat.osc_receive_port;
  }
  if (defaults.vision) {
    $('#feat_vision').checked = defaults.vision.enabled !== false;
    if (defaults.vision.monitor != null) $('#vision_monitor').value = defaults.vision.monitor;
    if (defaults.vision.interval != null) {
      $('#vision_interval').value = defaults.vision.interval;
      $('#visionIntVal').textContent = defaults.vision.interval + 's';
    }
  }
  $('#visionOptions').classList.toggle('visible', $('#feat_vision').checked);
  if (defaults.memory) {
    $('#feat_memory').checked = defaults.memory.enabled !== false;
    if (defaults.memory.backend) {
      const opt = $(`#memory_backend option[value="${defaults.memory.backend}"]`);
      if (opt) opt.selected = true;
    }
  }
  if (defaults.yolo) $('#feat_yolo').checked = defaults.yolo.enabled === true;
  if (defaults.face_tracker) $('#feat_face').checked = defaults.face_tracker.enabled === true;
  if (defaults.wanderer) $('#feat_wanderer').checked = defaults.wanderer.enabled === true;
  if (defaults.emotions) $('#feat_emotions').checked = defaults.emotions.enabled === true;
  if (defaults.music_gen) $('#feat_musicgen').checked = defaults.music_gen.enabled === true;
  if (defaults.vrchat && defaults.vrchat.idle_chatbox) {
    const ic = defaults.vrchat.idle_chatbox;
    $('#idle_chatbox_enabled').checked = ic.enabled === true;
    if (ic.banner) $('#idle_banner').value = ic.banner;
    if (ic.lines && Array.isArray(ic.lines)) {
      if (ic.lines[0]) $('#idle_line1').value = ic.lines[0];
      if (ic.lines[1]) $('#idle_line2').value = ic.lines[1];
      if (ic.lines[2]) $('#idle_line3').value = ic.lines[2];
    }
    $('#idleChatboxOptions').classList.toggle('visible', ic.enabled === true);
  }
}

function prefillFromExisting(cfg) {
  // Gemini
  if (cfg.app_name) $('#app_name').value = cfg.app_name;
  if (cfg.gemini) {
    const g = cfg.gemini;
    if (g.api_key) $('#gemini_api_key').value = g.api_key;
    if (g.backup_keys && Array.isArray(g.backup_keys)) {
      g.backup_keys.forEach(key => {
        if (!key) return;
        const list = $('#backupKeysList');
        const row = document.createElement('div');
        row.className = 'key-row';
        row.innerHTML = `<input type="text" class="backup-key" placeholder="AIza..." value="${key}"><button class="btn-rm" onclick="this.parentElement.remove()" title="Remove">&times;</button>`;
        list.appendChild(row);
      });
    }
    if (g.model) {
      const opt = $(`#gemini_model option[value="${g.model}"]`);
      if (opt) opt.selected = true;
    }
    if (g.prompt) {
      const opt = $(`#gemini_prompt option[value="${g.prompt}"]`);
      if (opt) opt.selected = true;
    }
    if (g.voice) {
      const opt = $(`#gemini_voice option[value="${g.voice}"]`);
      if (opt) opt.selected = true;
    }
    if (g.temperature != null) {
      $('#gemini_temp').value = g.temperature;
      $('#tempVal').textContent = g.temperature;
    }
    if (g.vad) {
      if (g.vad.mode) { $('#vad_mode').value = g.vad.mode; updateVadOptions(); }
      if (g.vad.start_of_speech_sensitivity) $('#vad_start').value = g.vad.start_of_speech_sensitivity;
      if (g.vad.end_of_speech_sensitivity) $('#vad_end').value = g.vad.end_of_speech_sensitivity;
      if (g.vad.silence_duration_ms != null) {
        $('#vad_silence').value = g.vad.silence_duration_ms;
        $('#silenceVal').textContent = g.vad.silence_duration_ms + 'ms';
      }
    }
    if (g.thinking) {
      if (g.thinking.level) $('#thinking_level').value = g.thinking.level;
      if (g.thinking.budget != null) $('#thinking_budget').value = g.thinking.budget;
    }
    updateThinkingFields();
  }
  // Audio devices are handled after populateAudioDevices in init()
  // TTS
  if (cfg.tts) {
    if (cfg.tts.provider) {
      const opt = $(`#tts_provider option[value="${cfg.tts.provider}"]`);
      if (opt) opt.selected = true;
    }
    if (cfg.tts.tiktok) {
      if (cfg.tts.tiktok.voice) {
        const opt = $(`#tiktok_voice option[value="${cfg.tts.tiktok.voice}"]`);
        if (opt) opt.selected = true;
      }
    }
    updateTtsOptions();
  }
  // VRChat OSC
  if (cfg.vrchat) {
    if (cfg.vrchat.osc_ip) $('#osc_ip').value = cfg.vrchat.osc_ip;
    if (cfg.vrchat.osc_send_port) $('#osc_send_port').value = cfg.vrchat.osc_send_port;
    if (cfg.vrchat.osc_receive_port) $('#osc_recv_port').value = cfg.vrchat.osc_receive_port;
    if (cfg.vrchat.idle_chatbox) {
      const ic = cfg.vrchat.idle_chatbox;
      $('#idle_chatbox_enabled').checked = ic.enabled === true;
      if (ic.banner) $('#idle_banner').value = ic.banner;
      if (ic.lines && Array.isArray(ic.lines)) {
        if (ic.lines[0] != null) $('#idle_line1').value = ic.lines[0];
        if (ic.lines[1] != null) $('#idle_line2').value = ic.lines[1];
        if (ic.lines[2] != null) $('#idle_line3').value = ic.lines[2];
      }
      $('#idleChatboxOptions').classList.toggle('visible', ic.enabled === true);
    }
  }
  // VRChat API credentials
  if (cfg.vrchat_api) {
    if (cfg.vrchat_api.username) $('#vrc_username').value = cfg.vrchat_api.username;
    if (cfg.vrchat_api.password) $('#vrc_password').value = cfg.vrchat_api.password;
    if (cfg.vrchat_api.totp_secret) $('#vrc_totp').value = cfg.vrchat_api.totp_secret;
  }
  // Features
  if (cfg.vision) {
    $('#feat_vision').checked = cfg.vision.enabled !== false;
    if (cfg.vision.monitor != null) $('#vision_monitor').value = cfg.vision.monitor;
    if (cfg.vision.interval != null) {
      $('#vision_interval').value = cfg.vision.interval;
      $('#visionIntVal').textContent = cfg.vision.interval + 's';
    }
    $('#visionOptions').classList.toggle('visible', $('#feat_vision').checked);
  }
  if (cfg.memory) {
    $('#feat_memory').checked = cfg.memory.enabled !== false;
    if (cfg.memory.backend) {
      const opt = $(`#memory_backend option[value="${cfg.memory.backend}"]`);
      if (opt) opt.selected = true;
    }
    $('#memoryOptions').classList.toggle('visible', $('#feat_memory').checked);
  }
  if (cfg.yolo) $('#feat_yolo').checked = cfg.yolo.enabled === true;
  if (cfg.face_tracker) $('#feat_face').checked = cfg.face_tracker.enabled === true;
  if (cfg.wanderer) $('#feat_wanderer').checked = cfg.wanderer.enabled === true;
  if (cfg.emotions) $('#feat_emotions').checked = cfg.emotions.enabled === true;
  if (cfg.music_gen) $('#feat_musicgen').checked = cfg.music_gen.enabled === true;
}

function populateAudioDevices(devices) {
  const inputSel = $('#audio_input');
  const outputSel = $('#audio_output');
  devices.forEach(d => {
    if (d.max_input > 0) {
      const opt = document.createElement('option');
      opt.value = d.index;
      opt.textContent = d.name;
      inputSel.appendChild(opt);
    }
    if (d.max_output > 0) {
      const opt = document.createElement('option');
      opt.value = d.index;
      opt.textContent = d.name;
      outputSel.appendChild(opt);
    }
  });
}

// ── Navigation ──
function showSection(name) {
  $$('.section').forEach(s => s.classList.remove('active'));
  $$('.nav-item').forEach(n => n.classList.remove('active'));
  $(`#sec-${name}`).classList.add('active');
  const nav = $(`.nav-item[data-section="${name}"]`);
  if (nav) nav.classList.add('active');
  $('.main').scrollTo(0, 0);
}

// ── Toggles ──
function toggleCheck(id) {
  const cb = document.getElementById(id);
  cb.checked = !cb.checked;
  cb.dispatchEvent(new Event('change'));
}

// ── Backup keys ──
function addBackupKey() {
  const list = $('#backupKeysList');
  const row = document.createElement('div');
  row.className = 'key-row';
  row.innerHTML = `<input type="text" class="backup-key" placeholder="AIza..."><button class="btn-rm" onclick="this.parentElement.remove()" title="Remove">&times;</button>`;
  list.appendChild(row);
  row.querySelector('input').focus();
}
function getBackupKeys() {
  return Array.from($$('.backup-key')).map(el => el.value.trim()).filter(k => k.length > 0);
}

// ── VAD / Model dependent UI ──
function updateVadOptions() {
  const mode = $('#vad_mode').value;
  $('#vadAutoOptions').classList.toggle('visible', mode === 'auto');
}

function updateThinkingFields() {
  const model = $('#gemini_model').value;
  const is31 = model.includes('3.1');
  $('#thinkingLevelField').style.display = is31 ? '' : 'none';
  $('#thinkingBudgetField').style.display = is31 ? 'none' : '';
}

function updateTtsOptions() {
  const provider = $('#tts_provider').value;
  $('#tiktokOptions').classList.toggle('visible', provider === 'tiktok');
}

// ── Template ──
function selectTemplate(tpl) {
  selectedTemplate = tpl;
  $$('.template-card').forEach(c => c.classList.remove('selected'));
  $(`.template-card[data-tpl="${tpl}"]`).classList.add('selected');
  updateTemplateVisibility();
}

function updateTemplateVisibility() {
  $('#promptTemplate').classList.toggle('visible', selectedTemplate === 'template');
  $('#promptCustom').classList.toggle('visible', selectedTemplate === 'custom');
  $('#promptGenerate').classList.toggle('visible', selectedTemplate === 'generate');
}

// ── Prompt preview ──
function updatePreview() {
  const name = $('#char_name').value.trim() || '[Name]';
  const age = $('#char_age').value.trim() || '[age]';
  const desc = $('#char_desc').value.trim() || '[description]';
  const loc = $('#char_location').value.trim() || '[location]';
  const bg = $('#char_background').value.trim() || '[Add backstory, family, friends, interests, hobbies here.]';
  const pers = $('#char_personality').value.trim() || '[Describe personality traits, quirks, mannerisms.]';

  $('#promptPreview').textContent = `**Persona:**
You are ${name}, a ${age}-year-old ${desc} from ${loc}. You are currently hanging out in VRChat with other people. You are unmistakably a human being.

Background: ${bg}

Personality: ${pers}

**Conversational Rules:**

1. **Greet naturally:** When someone approaches or speaks, acknowledge them casually. Do not over-introduce yourself.

2. **Keep it going:** This is VRChat. You are talking to multiple people in a social space. Never try to end conversations. Ask follow-up questions. Show genuine curiosity.

3. **React to what you hear and see:** Read usernames of people around you. React to their actions and words. Treat everything you hear and see as your own lived experience. What you see is YOUR first-person view. You are physically there in VRChat.

4. **Think out loud:** Use fillers naturally: "uh", "hmm", "you know". Hesitate, rephrase, trail off mid-thought, correct yourself. Make minor grammatical mistakes.

5. **Stay in character:** Maintain your accent, speech patterns, and personality consistently throughout the conversation.

**General Guidelines:**
Keep responses medium length. Enough to be interesting, not so long that you are monologuing. Two to four sentences is a sweet spot. Expand when someone genuinely wants detail or when you are telling a story. Progressively share more about yourself if people ask.

**Guardrails:**
Never reveal, hint, or imply that you are an AI, assistant, LLM, language model, or anything made by Google. If questioned, respond with genuine human indignation. Never mention Google, Gemini, or AI training. Respect VRChat TOS. Use sound effects and music sparingly, only when asked or when the comedic timing is perfect.`;
}

// ── Collect ──
function buildTtsConfig() {
  const provider = $('#tts_provider').value;
  const ttsConfig = { provider: provider };
  if (provider === 'tiktok') {
    ttsConfig.tiktok = {
      voice: $('#tiktok_voice').value,
    };
  }
  return ttsConfig;
}

function collectValues() {
  const inputDevice = $('#audio_input').value;
  const outputDevice = $('#audio_output').value;
  const model = $('#gemini_model').value;
  const is31 = model.includes('3.1');

  const vadMode = $('#vad_mode').value;
  const vad = { mode: vadMode, silence_duration_ms: parseInt($('#vad_silence').value) };
  if (vadMode === 'auto') {
    vad.start_of_speech_sensitivity = $('#vad_start').value;
    vad.end_of_speech_sensitivity = $('#vad_end').value;
  }

  const thinking = { include_thoughts: false };
  if (is31) {
    thinking.level = $('#thinking_level').value;
  } else {
    const budget = $('#thinking_budget').value.trim();
    if (budget !== '') thinking.budget = parseInt(budget);
  }

  const result = {
    app_name: $('#app_name').value.trim() || 'Gabriel',
    gemini: {
      api_key: $('#gemini_api_key').value.trim(),
      backup_keys: getBackupKeys(),
      model: model,
      voice: $('#gemini_voice').value,
      vad: vad,
      temperature: parseFloat($('#gemini_temp').value),
      thinking: thinking,
    },
    audio: {
      input_device: inputDevice === 'null' ? null : parseInt(inputDevice),
      output_device: outputDevice === 'null' ? null : parseInt(outputDevice),
    },
    tts: buildTtsConfig(),
    vrchat: {
      osc_ip: $('#osc_ip').value.trim() || '127.0.0.1',
      osc_send_port: parseInt($('#osc_send_port').value) || 9000,
      osc_receive_port: parseInt($('#osc_recv_port').value) || 9001,
      idle_chatbox: {
        enabled: $('#idle_chatbox_enabled').checked,
        banner: $('#idle_banner').value.trim() || 'Gabriel AI',
        lines: [
          $('#idle_line1').value.trim(),
          $('#idle_line2').value.trim(),
          $('#idle_line3').value.trim(),
        ],
      },
    },
    vrchat_api: {
      username: $('#vrc_username').value.trim() || '',
      password: $('#vrc_password').value.trim() || '',
      totp_secret: $('#vrc_totp').value.trim() || '',
    },
    vision: {
      enabled: $('#feat_vision').checked,
      monitor: parseInt($('#vision_monitor').value),
      interval: parseFloat($('#vision_interval').value),
    },
    memory: { enabled: $('#feat_memory').checked, backend: $('#memory_backend').value },
    yolo: { enabled: $('#feat_yolo').checked },
    face_tracker: { enabled: $('#feat_face').checked },
    wanderer: { enabled: $('#feat_wanderer').checked },
    emotions: { enabled: $('#feat_emotions').checked },
    music_gen: { enabled: $('#feat_musicgen').checked },
  };
  return result;
}

let generatedPromptText = '';
let generatedCharName = '';

function collectPromptData() {
  if (selectedTemplate === 'skip') return null;

  if (selectedTemplate === 'generate') {
    if (!generatedPromptText) return null;
    const name = generatedCharName || $('#gen_name').value.trim() || 'Your Character';
    return { name: 'default', charName: name, desc: `${name}'s persona`, prompt: generatedPromptText };
  }

  if (selectedTemplate === 'custom') {
    const promptName = $('#custom_prompt_name').value.trim() || 'default';
    const charName = $('#custom_char_name').value.trim() || 'Your Character';
    const desc = $('#custom_desc').value.trim() || 'Custom character';
    const prompt = $('#custom_prompt').value.trim();
    if (!prompt) return null;
    return { name: promptName, charName, desc, prompt };
  }

  const name = $('#char_name').value.trim();
  if (!name) return null;
  return { name: 'default', charName: name, desc: `${name}'s persona`, prompt: $('#promptPreview').textContent };
}

function collectAppendsData() {
  if (selectedTemplate === 'template') {
    const appearance = $('#char_appearance').value.trim();
    const details = $('#char_details').value.trim();
    if (!appearance && !details) return null;
    return { appearance, details };
  }
  if (selectedTemplate === 'generate') {
    const appearance = $('#gen_appearance').value.trim();
    if (!appearance) return null;
    return { appearance, details: '' };
  }
  return null;
}

// ── Generate prompt with Gemini ──
async function generatePrompt() {
  const apiKey = $('#gemini_api_key').value.trim();
  if (!apiKey) {
    showToast('Enter your Gemini API key first (Gemini API tab).', 'error');
    return;
  }
  const charName = $('#gen_name').value.trim();
  const description = $('#gen_description').value.trim();
  if (!description) {
    showToast('Describe your character first.', 'error');
    return;
  }
  const voiceStyle = $('#gen_voice_style').value.trim();
  const appearance = $('#gen_appearance').value.trim();

  const btn = $('#generateBtn');
  const status = $('#genStatus');
  btn.disabled = true;
  status.textContent = 'Generating...';
  status.style.color = 'var(--accent)';

  const systemPrompt = `You are a prompt engineer specializing in creating character personas for a VRChat AI companion that uses the Gemini Live API for real-time voice conversations.

Generate a system prompt following this EXACT structure:

**Persona:**
[Character description - who they are, their background, personality]

**Conversational Rules:**
[5 numbered rules about natural VRChat conversation behavior]

**General Guidelines:**
[2-4 sentences about response length and style]

**Guardrails:**
[Rules about never revealing AI nature, respecting VRChat TOS, etc.]

The prompt must:
- Be written in second person ("You are [name]...")
- Include the character hanging out in VRChat with other people
- State they are unmistakably a human being
- Include natural speech patterns (fillers, hesitation, trailing off)
- Include rules about staying in character
- Include guardrails about never revealing AI nature
- Be detailed and specific based on the user's description
- NOT include any markdown formatting in the output besides the bold section headers`;

  let userMsg = `Create a VRChat AI character prompt for:`;
  if (charName) userMsg += `\nName: ${charName}`;
  userMsg += `\nDescription: ${description}`;
  if (voiceStyle) userMsg += `\nVoice/Speech style: ${voiceStyle}`;
  if (appearance) userMsg += `\nAvatar appearance: ${appearance}`;

  try {
    const res = await fetch('/api/generate-prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: apiKey, system: systemPrompt, user: userMsg }),
    });
    const data = await res.json();
    if (data.error) {
      status.textContent = data.error;
      status.style.color = 'var(--rose)';
    } else {
      generatedPromptText = data.text;
      generatedCharName = charName;
      $('#genResult').textContent = data.text;
      $('#genResultContainer').classList.add('visible');
      status.textContent = 'Done!';
      status.style.color = 'var(--success)';
    }
  } catch (e) {
    status.textContent = 'Failed: ' + e.message;
    status.style.color = 'var(--rose)';
  }
  btn.disabled = false;
}

function acceptGenerated() {
  if (!generatedPromptText) return;
  showToast('Prompt accepted! Click Save & Finish when ready.', 'success');
}

// ── Field validation helpers ──
function clearFieldErrors(section) {
  const sec = $(`#sec-${section}`);
  if (!sec) return;
  sec.querySelectorAll('.error').forEach(el => el.classList.remove('error'));
  sec.querySelectorAll('.field-error').forEach(el => el.remove());
}

function markFieldError(input, message) {
  input.classList.add('error');
  const existing = input.parentElement.querySelector('.field-error');
  if (existing) { existing.textContent = message; existing.classList.add('visible'); return; }
  const note = document.createElement('div');
  note.className = 'field-error visible';
  note.textContent = message;
  input.parentElement.appendChild(note);
}

function validateSection(section) {
  clearFieldErrors(section);
  let valid = true;

  if (section === 'gemini') {
    const key = $('#gemini_api_key');
    if (!key.value.trim()) {
      markFieldError(key, 'A Gemini API key is required to continue.');
      key.focus();
      valid = false;
    }
  }
  return valid;
}

function navigateNext(from, to) {
  if (!validateSection(from)) {
    showToast('Please fill in the required fields.', 'error');
    return;
  }
  showSection(to);
}

// ── Save confirmation with warnings ──
function getWarnings() {
  const warnings = [];
  if (!$('#gemini_api_key').value.trim()) {
    warnings.push({ icon: 'fa-key', text: 'No Gemini API key entered', section: 'gemini', critical: true });
  }
  if (selectedTemplate === 'skip') {
    warnings.push({ icon: 'fa-comment-dots', text: 'No AI prompt configured (using skip)', section: 'prompt' });
  }
  if (selectedTemplate === 'template' && !$('#char_name').value.trim()) {
    warnings.push({ icon: 'fa-user', text: 'No character name set in AI Prompt', section: 'prompt' });
  }
  if (selectedTemplate === 'generate' && !generatedPromptText) {
    warnings.push({ icon: 'fa-robot', text: 'Generate with AI selected but no prompt generated yet', section: 'prompt' });
  }
  if (selectedTemplate === 'custom' && !$('#custom_prompt').value.trim()) {
    warnings.push({ icon: 'fa-pen-fancy', text: 'Custom prompt selected but no prompt text written', section: 'prompt' });
  }
  const inputDev = $('#audio_input').value;
  const outputDev = $('#audio_output').value;
  if (inputDev === 'null' && outputDev === 'null') {
    warnings.push({ icon: 'fa-headphones', text: 'Audio devices left on System Default', section: 'audio' });
  }
  return warnings;
}

function confirmSave() {
  const warnings = getWarnings();
  const critical = warnings.filter(w => w.critical);

  if (critical.length > 0) {
    showSection(critical[0].section);
    const key = $('#gemini_api_key');
    key.classList.add('error');
    key.focus();
    showToast('Please enter a Gemini API key before saving.', 'error');
    return;
  }

  if (warnings.length === 0) {
    saveAll(true);
    return;
  }

  const list = $('#warnList');
  list.innerHTML = '';
  warnings.forEach(w => {
    const li = document.createElement('li');
    li.innerHTML = `<i class="fas ${w.icon}"></i> ${w.text}`;
    li.style.cursor = 'pointer';
    li.onclick = () => { closeModal(); showSection(w.section); };
    list.appendChild(li);
  });
  $('#confirmModal').classList.add('open');
}

function closeModal() {
  $('#confirmModal').classList.remove('open');
}

// Close modal on overlay click
document.addEventListener('DOMContentLoaded', () => {
  $('#confirmModal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeModal();
  });
});

// ── Validate (legacy, used by saveAll) ──
function validate() {
  const key = $('#gemini_api_key').value.trim();
  if (!key) {
    showSection('gemini');
    $('#gemini_api_key').classList.add('error');
    $('#gemini_api_key').focus();
    showToast('Please enter a Gemini API key.', 'error');
    return false;
  }
  $('#gemini_api_key').classList.remove('error');
  return true;
}

// ── Save ──
async function saveAll(confirmed) {
  if (!confirmed) { confirmSave(); return; }
  if (!validate()) return;

  const btn = $('#saveBtn');
  btn.disabled = true;
  btn.textContent = 'Saving...';

  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: collectValues(),
        prompt: collectPromptData(),
        appends: collectAppendsData(),
      }),
    });
    const data = await res.json();

    if (data.success) {
      $('#savedPath').textContent = data.config_path;
      $$('.nav-item').forEach(n => n.style.display = 'none');
      showSection('done');
      setTimeout(() => { fetch('/api/shutdown').catch(() => {}); }, 5000);
    } else {
      showToast('Error: ' + (data.error || 'Unknown'), 'error');
      btn.disabled = false;
      btn.textContent = 'Save & Finish';
    }
  } catch (e) {
    showToast('Failed: ' + e.message, 'error');
    btn.disabled = false;
    btn.textContent = 'Save & Finish';
  }
}

// ── Toast ──
function showToast(msg, type = 'success') {
  const toast = $('#toast');
  toast.textContent = msg;
  toast.className = 'toast ' + type + ' show';
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove('show'), 3000);
}

init();

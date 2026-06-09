const API_BASE = (() => {
  const { protocol, hostname, port, origin } = window.location;
  const isDevFrontend = ['5173', '5174'].includes(port);
  const isLocalHost = ['localhost', '127.0.0.1', '0.0.0.0'].includes(hostname);
  const isIpv4Host = /^\d+\.\d+\.\d+\.\d+$/.test(hostname);
  if (port === '8001') return `${origin}/api`;
  if (isDevFrontend || isLocalHost || isIpv4Host) return `${protocol}//${hostname}:8001/api`;
  return `${origin}/api`;
})();

const urlParams = new URLSearchParams(window.location.search);
let projectId = urlParams.get('id');

const DEFAULT_AP_MODELS = [
  {
    id: 'u6_pro',
    vendor: 'Ubiquiti',
    model: 'UniFi 6 Pro',
    gain24: 4,
    gain5: 6,
    maxPower: 26,
    defaultPower24: 20,
    defaultPower5: 18,
    rate24: 573,
    rate5: 4800,
    mimo: '4 x 4 (DL/UL MU-MIMO)',
    clients: 300,
    planningClients: 50,
    coverage: 'Large office / classroom',
    bssids: '8 per Radio',
    standards: '802.11ax (WiFi 6), 802.11ac (WiFi 5), 802.11n',
    color: '#3b82f6',
    image: 'assets/unifi6_pro.png'
  },
  {
    id: 'u6_lite',
    vendor: 'Ubiquiti',
    model: 'UniFi 6 Lite',
    gain24: 2.8,
    gain5: 3,
    maxPower: 23,
    defaultPower24: 17,
    defaultPower5: 16,
    rate24: 300,
    rate5: 1200,
    mimo: '2 x 2 (DL/UL MU-MIMO)',
    clients: 300,
    planningClients: 30,
    coverage: 'Small office / home',
    bssids: '8 per Radio',
    standards: '802.11ax (WiFi 6), 802.11ac (WiFi 5), 802.11n',
    color: '#60a5fa',
    image: 'assets/unifi6_lite.png'
  },
  {
    id: 'eap660',
    vendor: 'TP-Link',
    model: 'EAP660 HD',
    gain24: 4,
    gain5: 5,
    maxPower: 26,
    defaultPower24: 20,
    defaultPower5: 18,
    rate24: 1148,
    rate5: 2402,
    mimo: '4 x 4 (DL MU-MIMO)',
    clients: 1020,
    planningClients: 75,
    coverage: 'High-density indoor area',
    bssids: '16 total',
    standards: 'IEEE 802.11ax/ac/n/g/b/a',
    color: '#06b6d4',
    image: 'assets/TP-Link_EAP660HD.png'
  }
];

function cloneDefaultApModels() {
  return JSON.parse(JSON.stringify(DEFAULT_AP_MODELS));
}

function normalizeApModel(apModel, fallbackModel = {}) {
  const maxPower = Number(apModel.maxPower ?? fallbackModel.maxPower ?? 20);
  const defaultPower24 = Number(apModel.defaultPower24 ?? apModel.maxPower ?? fallbackModel.defaultPower24 ?? fallbackModel.maxPower ?? 18);
  const defaultPower5 = Number(apModel.defaultPower5 ?? apModel.maxPower ?? fallbackModel.defaultPower5 ?? fallbackModel.maxPower ?? 17);
  const bands = Array.isArray(apModel.bands)
    ? apModel.bands.map(String)
    : (Array.isArray(fallbackModel.bands) ? fallbackModel.bands.map(String) : ['2.4', '5']);
  const supports6 = bands.includes('6');
  const defaultPower6 = supports6
    ? clamp(Number(apModel.defaultPower6 ?? fallbackModel.defaultPower6 ?? apModel.maxPower ?? fallbackModel.maxPower ?? 18), 1, maxPower)
    : 0;
  return {
    id: apModel.id || fallbackModel.id || `ap_model_${Date.now()}`,
    vendor: apModel.vendor || fallbackModel.vendor || 'Custom',
    model: apModel.model || fallbackModel.model || 'Custom AP',
    class: apModel.class || fallbackModel.class || 'custom',
    bands,
    gain24: Number(apModel.gain24 ?? fallbackModel.gain24 ?? 2),
    gain5: Number(apModel.gain5 ?? fallbackModel.gain5 ?? 2),
    gain6: Number(apModel.gain6 ?? fallbackModel.gain6 ?? 0),
    maxPower,
    defaultPower24: clamp(defaultPower24, 1, maxPower),
    defaultPower5: clamp(defaultPower5, 1, maxPower),
    defaultPower6,
    rate24: Number(apModel.rate24 ?? fallbackModel.rate24 ?? 300),
    rate5: Number(apModel.rate5 ?? fallbackModel.rate5 ?? 866),
    rate6: Number(apModel.rate6 ?? fallbackModel.rate6 ?? 0),
    mimo: apModel.mimo || fallbackModel.mimo || '2 x 2 MIMO',
    clients: Number(apModel.clients ?? fallbackModel.clients ?? 100),
    recommendedClients: Number(apModel.recommendedClients ?? fallbackModel.recommendedClients ?? apModel.clients ?? fallbackModel.clients ?? 80),
    planningClients: Number(apModel.planningClients ?? fallbackModel.planningClients ?? Math.min(apModel.recommendedClients ?? apModel.clients ?? 80, 40)),
    coverage: apModel.coverage || fallbackModel.coverage || 'Indoor',
    mounting: Array.isArray(apModel.mounting)
      ? apModel.mounting
      : (Array.isArray(fallbackModel.mounting) ? fallbackModel.mounting : ['ceiling']),
    cost: Number(apModel.cost ?? fallbackModel.cost ?? 100),
    powerDrawW: Number(apModel.powerDrawW ?? fallbackModel.powerDrawW ?? 12),
    bssids: apModel.bssids || fallbackModel.bssids || '8 total',
    standards: apModel.standards || fallbackModel.standards || '802.11ac/ax',
    launched: apModel.launched || fallbackModel.launched || '',
    color: apModel.color || fallbackModel.color || '#2563eb',
    image: apModel.image ?? fallbackModel.image ?? ''
  };
}

function normalizeApModelList(apModels) {
  return (apModels || []).map(apModel => {
    const fallback = DEFAULT_AP_MODELS.find(defaultModel => defaultModel.id === apModel.id) || {};
    return normalizeApModel(apModel, fallback);
  });
}

function mergeApModelLists(catalogModels, projectModels) {
  const merged = normalizeApModelList(catalogModels);
  const indexById = new Map(merged.map((model, index) => [model.id, index]));

  normalizeApModelList(projectModels).forEach(projectModel => {
    if (indexById.has(projectModel.id)) {
      merged[indexById.get(projectModel.id)] = normalizeApModel(projectModel, merged[indexById.get(projectModel.id)]);
    } else {
      indexById.set(projectModel.id, merged.length);
      merged.push(projectModel);
    }
  });

  return merged;
}

async function loadApCatalog() {
  try {
    const res = await fetch(`${API_BASE}/ap-catalog`);
    if (!res.ok) throw new Error(`AP catalog request failed with ${res.status}`);
    const catalog = await res.json();
    const models = Array.isArray(catalog.models) ? catalog.models : (Array.isArray(catalog) ? catalog : []);
    if (models.length === 0) throw new Error('AP catalog did not contain models');

    state.apModels = normalizeApModelList(models);
    if (!state.apModels.some(model => model.id === state.selectedApModel)) {
      state.selectedApModel = state.apModels[0]?.id || 'u6_pro';
    }
  } catch (err) {
    console.warn('Using built-in AP catalog fallback:', err);
    state.apModels = cloneDefaultApModels();
  }
}

// Pre-load AP images for canvas rendering
const apImageCache = {};
function preloadApImages() {
  Object.keys(apImageCache).forEach(key => delete apImageCache[key]);
  state.apModels.forEach(model => {
    if (!model.image) return;
    const img = new Image();
    img.src = model.image;
    img.onload = () => {
      apImageCache[model.id] = img;
      draw(); // Redraw once image is loaded
    };
  });
}

// Initialize workspace
window.addEventListener('load', initializeWorkspace);

async function initializeWorkspace() {
  await loadApCatalog();
  if (projectId) {
    await fetchProject(projectId);
  } else {
    preloadApImages();
    renderDevicePanel();
  }
}

async function fetchProject(id) {
  try {
    const res = await fetch(`${API_BASE}/projects/${id}`);
    const data = await res.json();

    const topBarName = document.getElementById('topBarProjectName');
    if (topBarName) topBarName.innerText = data.name || "Project Workspace";

    state.elements = data.elements || [];
    state.accessPoints = data.access_points || [];
    if (data.parameters) {
      state.selectedFrequency = normalizeFrequencyGHz(data.parameters.selectedFrequency);
      if (Array.isArray(data.parameters.wallTypes) && data.parameters.wallTypes.length > 0) {
        state.wallTypes = data.parameters.wallTypes;
      }
      if (data.parameters.selectedWallType && state.wallTypes.some(wall => wall.id === data.parameters.selectedWallType)) {
        state.selectedWallType = data.parameters.selectedWallType;
      }
      if (data.parameters.floorHeight) {
        state.floorHeight = Number(data.parameters.floorHeight);
        const floorHeightText = document.getElementById('floorHeightText');
        if (floorHeightText) floorHeightText.innerText = `${state.floorHeight}m`;
      }
      state.rooms = normalizeRooms(Array.isArray(data.parameters.rooms) ? data.parameters.rooms : []);
      state.serviceBoundary = Array.isArray(data.parameters.serviceBoundary) ? data.parameters.serviceBoundary : [];
      state.doorOpenings = Array.isArray(data.parameters.doorOpenings) ? data.parameters.doorOpenings : [];
      state.showRoomsOverlay = data.parameters.showRoomsOverlay ?? state.rooms.length > 0;
      state.autoPlacementLocked = data.parameters.autoPlacementLocked ?? state.rooms.length > 0;
      state.apPlacementTab = data.parameters.apPlacementTab || (state.autoPlacementLocked ? 'auto' : 'manual');
      state.manualAccessPointsBackup = normalizeAccessPoints(data.parameters.manualAccessPointsBackup || []);
      state.autoAccessPointsBackup = normalizeAccessPoints(data.parameters.autoAccessPointsBackup || []);
      if (Array.isArray(data.parameters.apModels) && data.parameters.apModels.length > 0) {
        state.apModels = mergeApModelLists(state.apModels, data.parameters.apModels);
      }
    }
    state.accessPoints = normalizeAccessPoints(state.accessPoints);
    preloadApImages();
    syncAccessPointFrequencies();
    syncFrequencyButtons();
    syncLegendUI();
    renderRoomsPanel();
    if (data.parameters && data.parameters.blueprintImageSrc) {
      state.blueprintImageSrc = data.parameters.blueprintImageSrc;
      const img = new Image();
      img.onload = () => {
        state.blueprintImgObj = img;
        document.getElementById('workspaceEmptyState').style.display = 'none';
        draw();
      };
      img.src = state.blueprintImageSrc;
      if (data.parameters.pixelsPerMeter) {
        state.pixelsPerMeter = data.parameters.pixelsPerMeter;
        state.isScaleSet = true;
      }
    } else {
      document.getElementById('workspaceEmptyState').style.display = 'flex';
    }
    invalidateHeatmap();
    saveHistory();
    draw();
  } catch (err) {
    console.error("Failed to fetch project", err);
  }
}

// State
let state = {
  mode: 'select',
  isScaleSet: false,
  elements: [],
  rooms: [],
  serviceBoundary: [],
  doorOpenings: [],
  showRoomsOverlay: true,
  apPlacementTab: 'manual',
  autoPlacementLocked: false,
  manualAccessPointsBackup: [],
  autoAccessPointsBackup: [],
  selectedRoomIndex: null,
  accessPoints: [],
  isDrawing: false,
  currentLine: null,
  transform: { x: 0, y: 0, scale: 1 },
  hoveredAp: null,
  draggingAp: null,
  hoveredWall: null,
  hoveredHandle: null,
  draggingHandle: null,
  isDragging: false,
  draggingWall: null,
  lastMousePos: null,
  selectedElementIndex: null,
  selectedElementType: null, // 'wall' or 'ap'
  heatmapImageSrc: null,
  blueprintImageSrc: null,
  blueprintImgObj: null,
  wallTypes: [
    { id: 'common_brick', name: 'Common brick wall', db24: 10, db5: 15, db6: 17, thickness: 120, color: '#f87171' },
    { id: 'thick_brick', name: 'Thick brick wall', db24: 15, db5: 25, db6: 28, thickness: 240, color: '#dc2626' },
    { id: 'concrete', name: 'Concrete', db24: 25, db5: 30, db6: 33, thickness: 240, color: '#f59e0b' },
    { id: 'gypsum', name: 'Gypsum board', db24: 3, db5: 4, db6: 5, thickness: 8, color: '#a78bfa' },
    { id: 'foam', name: 'Foam materials', db24: 3, db5: 4, db6: 5, thickness: 8, color: '#f472b6' },
    { id: 'hollow_wood', name: 'Hollow wood', db24: 2, db5: 3, db6: 4, thickness: 20, color: '#fdba74' },
    { id: 'common_wood_door', name: 'Common wooden door', db24: 3, db5: 4, db6: 5, thickness: 40, color: '#fb923c' },
    { id: 'solid_wood_door', name: 'Solid wood door', db24: 10, db5: 15, db6: 17, thickness: 40, color: '#c2410c' },
    { id: 'common_glass', name: 'Common glass', db24: 4, db5: 7, db6: 8, thickness: 8, color: '#67e8f9' },
    { id: 'thick_glass', name: 'Thick glass', db24: 8, db5: 10, db6: 12, thickness: 12, color: '#06b6d4' },
    { id: 'armored_glass', name: 'Armored glass', db24: 25, db5: 35, db6: 40, thickness: 30, color: '#0891b2' },
    { id: 'load_bearing_column', name: 'Load-bearing column', db24: 25, db5: 30, db6: 33, thickness: 500, color: '#84cc16' },
    { id: 'shutter_door', name: 'Shutter door', db24: 15, db5: 20, db6: 23, thickness: 10, color: '#10b981' },
    { id: 'color_steel_panel', name: 'Color steel sandwich panel', db24: 30, db5: 35, db6: 40, thickness: 80, color: '#3b82f6' },
    { id: 'elevator', name: 'Elevator', db24: 30, db5: 35, db6: 40, thickness: 80, color: '#7e22ce' },
    { id: 'shelf', name: 'Shelf', db24: 15, db5: 20, db6: 23, thickness: 80, color: '#eab308' }
  ],
  selectedWallType: 'common_brick',
  apModels: cloneDefaultApModels(),
  selectedApModel: 'u6_pro',
  selectedFrequency: 2.4,
  pixelsPerMeter: 30.7, // Default: 1m = 30.7px
  floorHeight: 3.0,
  iconSize: 1.0,
  history: [],
  historyIndex: -1
};

function saveHistory() {
  if (state.historyIndex < state.history.length - 1) {
    state.history = state.history.slice(0, state.historyIndex + 1);
  }
  state.history.push({
    elements: JSON.parse(JSON.stringify(state.elements)),
    rooms: JSON.parse(JSON.stringify(state.rooms || [])),
    serviceBoundary: JSON.parse(JSON.stringify(state.serviceBoundary || [])),
    showRoomsOverlay: state.showRoomsOverlay,
    autoPlacementLocked: state.autoPlacementLocked,
    apPlacementTab: state.apPlacementTab,
    manualAccessPointsBackup: JSON.parse(JSON.stringify(state.manualAccessPointsBackup || [])),
    autoAccessPointsBackup: JSON.parse(JSON.stringify(state.autoAccessPointsBackup || [])),
    accessPoints: JSON.parse(JSON.stringify(state.accessPoints)),
    apModels: JSON.parse(JSON.stringify(state.apModels))
  });
  state.historyIndex++;
}

function undo() {
  if (state.historyIndex > 0) {
    state.historyIndex--;
    const prevState = state.history[state.historyIndex];
    state.elements = JSON.parse(JSON.stringify(prevState.elements));
    state.rooms = normalizeRooms(JSON.parse(JSON.stringify(prevState.rooms || [])));
    state.serviceBoundary = JSON.parse(JSON.stringify(prevState.serviceBoundary || []));
    state.showRoomsOverlay = prevState.showRoomsOverlay ?? state.rooms.length > 0;
    state.autoPlacementLocked = prevState.autoPlacementLocked ?? state.rooms.length > 0;
    state.apPlacementTab = prevState.apPlacementTab || (state.autoPlacementLocked ? 'auto' : 'manual');
    state.manualAccessPointsBackup = normalizeAccessPoints(JSON.parse(JSON.stringify(prevState.manualAccessPointsBackup || [])));
    state.autoAccessPointsBackup = normalizeAccessPoints(JSON.parse(JSON.stringify(prevState.autoAccessPointsBackup || [])));
    state.accessPoints = normalizeAccessPoints(JSON.parse(JSON.stringify(prevState.accessPoints)));
    if (prevState.apModels) {
      state.apModels = normalizeApModelList(JSON.parse(JSON.stringify(prevState.apModels)));
      preloadApImages();
    }
    renderRoomsPanel();
    draw();
  } else if (state.historyIndex === 0) {
    state.historyIndex = -1;
    state.elements = [];
    state.rooms = [];
    state.serviceBoundary = [];
    state.showRoomsOverlay = true;
    state.autoPlacementLocked = false;
    state.apPlacementTab = 'manual';
    state.manualAccessPointsBackup = [];
    state.autoAccessPointsBackup = [];
    state.accessPoints = [];
    renderRoomsPanel();
    draw();
  }
}

function redo() {
  if (state.historyIndex < state.history.length - 1) {
    state.historyIndex++;
    const nextState = state.history[state.historyIndex];
    state.elements = JSON.parse(JSON.stringify(nextState.elements));
    state.rooms = normalizeRooms(JSON.parse(JSON.stringify(nextState.rooms || [])));
    state.serviceBoundary = JSON.parse(JSON.stringify(nextState.serviceBoundary || []));
    state.showRoomsOverlay = nextState.showRoomsOverlay ?? state.rooms.length > 0;
    state.autoPlacementLocked = nextState.autoPlacementLocked ?? state.rooms.length > 0;
    state.apPlacementTab = nextState.apPlacementTab || (state.autoPlacementLocked ? 'auto' : 'manual');
    state.manualAccessPointsBackup = normalizeAccessPoints(JSON.parse(JSON.stringify(nextState.manualAccessPointsBackup || [])));
    state.autoAccessPointsBackup = normalizeAccessPoints(JSON.parse(JSON.stringify(nextState.autoAccessPointsBackup || [])));
    state.accessPoints = normalizeAccessPoints(JSON.parse(JSON.stringify(nextState.accessPoints)));
    if (nextState.apModels) {
      state.apModels = normalizeApModelList(JSON.parse(JSON.stringify(nextState.apModels)));
      preloadApImages();
    }
    renderRoomsPanel();
    draw();
  }
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (state.isDrawing) {
      state.isDrawing = false;
      state.currentLine = null;
      draw();
    } else {
      setActiveTool('select', document.querySelector('.tool-btn[data-tool="select"]'));
    }
    return;
  }

  if (e.ctrlKey || e.metaKey) {
    if (e.key === 'z' || e.key === 'Z') {
      e.preventDefault();
      undo();
    } else if (e.key === 'y' || e.key === 'Y') {
      e.preventDefault();
      redo();
    }
  }
});

// DOM Elements
const canvas = document.getElementById('floorCanvas');
const ctx = canvas.getContext('2d');
const wrapper = document.getElementById('canvasWrapper');
const heatmapImage = document.getElementById('heatmapImage');
let pendingDrawFrame = null;

function requestDraw() {
  if (pendingDrawFrame !== null) return;
  pendingDrawFrame = requestAnimationFrame(() => {
    pendingDrawFrame = null;
    draw();
  });
}

function normalizeFrequencyGHz(value) {
  const numeric = Number(value);
  return numeric >= 4 ? 5 : 2.4;
}

function getActiveFrequencyGHz() {
  return normalizeFrequencyGHz(state.selectedFrequency);
}

function getActiveBandKey() {
  return getActiveFrequencyGHz() >= 5 ? '5ghz' : '2.4ghz';
}

function getActiveFrequencyLabel() {
  return getActiveFrequencyGHz() >= 5 ? '5 GHz' : '2.4 GHz';
}

function getApModelById(modelId) {
  return state.apModels.find(model => model.id === modelId) || state.apModels[0];
}

function getAntennaGainForBand(apModel, ap, frequencyGHz) {
  if (frequencyGHz >= 5) {
    return apModel?.gain5 ?? ap?.gain5 ?? apModel?.gain24 ?? ap?.gain24 ?? 0;
  }
  return apModel?.gain24 ?? ap?.gain24 ?? apModel?.gain5 ?? ap?.gain5 ?? 0;
}

function getDefaultTxPowerForBand(apModel, frequencyGHz) {
  if (frequencyGHz >= 5) {
    return Number(apModel?.defaultPower5 ?? apModel?.maxPower ?? 17);
  }
  return Number(apModel?.defaultPower24 ?? apModel?.maxPower ?? 18);
}

function getTxPowerForBand(apModel, ap, frequencyGHz) {
  if (frequencyGHz >= 5) {
    return Number(ap?.power5 ?? ap?.power ?? getDefaultTxPowerForBand(apModel, frequencyGHz));
  }
  return Number(ap?.power24 ?? ap?.power ?? getDefaultTxPowerForBand(apModel, frequencyGHz));
}

function getDefaultApHeightM() {
  return Math.max(0.1, Number(state.floorHeight || 3.0) - 0.2);
}

function getClientHeightM() {
  return 1.0;
}

function normalizeAccessPoint(ap, index = 0) {
  const apModel = getApModelById(ap.model_id || state.selectedApModel);
  const power24 = Number(ap.power24 ?? ap.power ?? getDefaultTxPowerForBand(apModel, 2.4));
  const power5 = Number(ap.power5 ?? ap.power ?? getDefaultTxPowerForBand(apModel, 5));
  const activeFrequency = normalizeFrequencyGHz(ap.frequency ?? state.selectedFrequency);
  const fallbackName = `AP ${index + 1}`;
  const name = typeof ap.name === 'string' && ap.name.trim() ? ap.name.trim() : fallbackName;

  return {
    ...ap,
    name,
    model_id: ap.model_id || apModel.id,
    model: ap.model || apModel.model,
    vendor: ap.vendor || apModel.vendor,
    gain24: Number(ap.gain24 ?? apModel.gain24),
    gain5: Number(ap.gain5 ?? apModel.gain5),
    power24,
    power5,
    z: Number(ap.z ?? getDefaultApHeightM()),
    frequency: activeFrequency,
    power: activeFrequency >= 5 ? power5 : power24
  };
}

function normalizeAccessPoints(accessPoints) {
  return (accessPoints || []).map((ap, index) => normalizeAccessPoint(ap, index));
}

// Estimate user demand from room area so AP placement matches real need:
// large rooms carry more users, small rooms few, tiny rooms ~none. Default
// density is ~1 active user per m² (e.g. 42 m² -> ~42, 7 m² -> ~7, 3 m² -> ~3).
function estimateRoomClients(areaM2, density = 1.0) {
  const area = Number(areaM2) || 0;
  if (area <= 0) return 0;
  return Math.max(0, Math.round(area * density));
}

// Priority scales with room area: large rooms High (3), medium Medium (2),
// small Low (1). Keeps big rooms prioritised for coverage/capacity.
function estimateRoomPriority(areaM2) {
  const area = Number(areaM2) || 0;
  if (area >= 30) return 3;
  if (area >= 15) return 2;
  return 1;
}

function normalizeRoom(room, index = 0) {
  const roomType = room.roomType || room.type || 'office';
  const isStairs = roomType === 'stairs';
  return {
    ...room,
    name: room.name || `Room ${index + 1}`,
    type: roomType,
    roomType,
    priority: isStairs ? 0 : Number(room.priority ?? 1),
    userDensity: Number(room.userDensity ?? 0.1),
    clients: isStairs ? 0 : Number(room.clients ?? Math.round(Number(room.userDensity ?? 0.1) * Number(room.areaM2 || 0) * 10)),
    coverageTarget: Number(room.coverageTarget ?? (isStairs ? -80 : -67)),
    excluded: isStairs || Boolean(room.excluded)
  };
}

function normalizeRooms(rooms) {
  return (rooms || []).map(normalizeRoom);
}

function applyModelToAccessPoints(modelId, overwritePower = true) {
  const apModel = getApModelById(modelId);
  state.accessPoints = state.accessPoints.map(ap => {
    if ((ap.model_id || state.selectedApModel) !== modelId) {
      return ap;
    }

    const nextAp = {
      ...ap,
      model_id: apModel.id,
      model: apModel.model,
      vendor: apModel.vendor,
      gain24: Number(apModel.gain24),
      gain5: Number(apModel.gain5)
    };

    if (overwritePower) {
      nextAp.power24 = getDefaultTxPowerForBand(apModel, 2.4);
      nextAp.power5 = getDefaultTxPowerForBand(apModel, 5);
    }

    return normalizeAccessPoint(nextAp);
  });
}

const BAND_SIMULATION_PROFILES = {
  '2.4ghz': {
    pathLossExponent: 3.0,
    fadeMarginDb: 7,
    wallLossMultiplier: 1.0,
    wallPenaltyRadiusM: 0,
    wallPenaltyDb: 0
  },
  '5ghz': {
    pathLossExponent: 3.1,
    fadeMarginDb: 8,
    wallLossMultiplier: 1.0,
    wallPenaltyRadiusM: 0,
    wallPenaltyDb: 0
  }
};

function getSimulationProfile(frequencyGHz) {
  return frequencyGHz >= 5 ? BAND_SIMULATION_PROFILES['5ghz'] : BAND_SIMULATION_PROFILES['2.4ghz'];
}

function syncAccessPointFrequencies() {
  const frequencyGHz = getActiveFrequencyGHz();
  state.accessPoints = state.accessPoints.map(ap => {
    const apModel = getApModelById(ap.model_id || state.selectedApModel);
    ap.frequency = frequencyGHz;
    ap.power = getTxPowerForBand(apModel, ap, frequencyGHz);
    return normalizeAccessPoint(ap);
  });
}

function syncFrequencyButtons() {
  document.querySelectorAll('.js-frequency-button').forEach(button => {
    const buttonFrequency = normalizeFrequencyGHz(button.dataset.frequency);
    button.classList.toggle('active', buttonFrequency === getActiveFrequencyGHz());
  });
}

function syncLegendUI() {
  const bandLabel = getActiveFrequencyLabel();
  const topLabel = document.getElementById('legendTopLabel');
  const bottomLabel = document.getElementById('legendBottomLabel');
  const tickGood = document.getElementById('legendTickGood');
  const tickMid = document.getElementById('legendTickMid');
  const tickPoor = document.getElementById('legendTickPoor');

  if (topLabel) topLabel.textContent = `Good`;
  if (bottomLabel) bottomLabel.innerHTML = `Poor<br><small>${bandLabel}</small>`;

  if (getActiveFrequencyGHz() >= 5) {
    if (tickGood) tickGood.textContent = '-62';
    if (tickMid) tickMid.textContent = '-77';
    if (tickPoor) tickPoor.textContent = '-87';
  } else {
    if (tickGood) tickGood.textContent = '-60';
    if (tickMid) tickMid.textContent = '-75';
    if (tickPoor) tickPoor.textContent = '-85';
  }
}

function getLegendSignalRange() {
  if (getActiveFrequencyGHz() >= 5) {
    return { good: -62, mid: -77, poor: -87 };
  }
  return { good: -60, mid: -75, poor: -85 };
}

function getSignalQualityLabel(dbm) {
  const range = getLegendSignalRange();
  if (dbm >= range.good) return 'Good';
  if (dbm >= range.mid) return 'Fair';
  if (dbm >= range.poor) return 'Poor';
  return 'No Signal';
}

function resetLegendHover() {
  const legend = document.querySelector('.legend');
  const marker = document.getElementById('legendHoverMarker');
  if (legend) legend.classList.remove('is-hovering');
  if (marker) marker.style.display = 'none';
  syncLegendUI();
}

function setActiveFrequency(frequencyGHz) {
  const normalizedFrequency = normalizeFrequencyGHz(frequencyGHz);
  if (normalizedFrequency === getActiveFrequencyGHz()) {
    syncFrequencyButtons();
    syncLegendUI();
    return;
  }

  state.selectedFrequency = normalizedFrequency;
  syncAccessPointFrequencies();
  syncFrequencyButtons();
  syncLegendUI();
  invalidateHeatmap();
  renderWallPanel();
  renderDevicePanel();
  draw();

  if (document.getElementById('deviceListModal')?.style.display === 'flex') {
    renderDeviceList();
  }
}

document.querySelectorAll('.js-frequency-button').forEach(button => {
  button.addEventListener('click', () => {
    setActiveFrequency(button.dataset.frequency);
  });
});
syncFrequencyButtons();
syncLegendUI();

document.getElementById('btnRefreshHeatmap')?.addEventListener('click', () => {
  invalidateHeatmap();
  draw();
});

// Resize canvas
function resize() {
  const rect = wrapper.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
  draw();
}
window.addEventListener('resize', resize);
setTimeout(resize, 100);

// Tools
async function setActiveTool(tool, buttonElement) {
  // Cancel current drawing if tool changes
  state.isDrawing = false;
  state.currentLine = null;
  draw();

  // Enforce scale setting before allowing walls or APs
  if ((tool === 'wall' || tool === 'ap') && !state.isScaleSet) {
    await customAlert("Please set the floor plan scale first using the 'Set Scale' tool to ensure accurate signal simulation.", "Scale Required");
    const scaleBtn = document.querySelector('.tool-btn[data-tool="scale"]');
    if (scaleBtn) setActiveTool('scale', scaleBtn);
    return;
  }

  state.mode = tool;
  document.querySelectorAll('.tool-btn, .view-controls button').forEach(b => b.classList.remove('active'));
  if (buttonElement) buttonElement.classList.add('active');

  // Panels
  document.getElementById('drawWallsPanel').style.display = tool === 'wall' ? 'flex' : 'none';
  document.getElementById('placeDevicesPanel').style.display = tool === 'ap' ? 'flex' : 'none';

  // Scale Banner
  const banner = document.getElementById('scaleBanner');
  if (banner) banner.style.display = tool === 'scale' ? 'flex' : 'none';

  if (tool === 'wall') renderWallPanel();
  if (tool === 'ap') {
    renderDevicePanel();
    renderAutoPlacementPanel();
    syncApPlacementTabs();
  }

  // Set cursor based on mode
  if (tool === 'pan') {
    canvas.style.cursor = 'grab';
  } else if (tool === 'select') {
    canvas.style.cursor = 'default';
  } else if (tool === 'ap' && state.apPlacementTab === 'auto') {
    canvas.style.cursor = 'default';
  } else {
    canvas.style.cursor = 'crosshair';
  }
}

document.getElementById('closeScaleBanner')?.addEventListener('click', () => {
  document.getElementById('scaleBanner').style.display = 'none';
  const wallBtn = document.querySelector('.tool-btn[data-tool="wall"]');
  if (wallBtn) setActiveTool('wall', wallBtn);
});

document.getElementById('btnEditFloorHeight')?.addEventListener('click', async () => {
  const newVal = await customPrompt("Enter Floor Height (meters):", state.floorHeight);
  if (newVal && !isNaN(newVal)) {
    const previousDefaultHeight = getDefaultApHeightM();
    state.floorHeight = Math.max(1, parseFloat(newVal));
    const nextDefaultHeight = getDefaultApHeightM();
    state.accessPoints = state.accessPoints.map(ap => {
      if (Math.abs(Number(ap.z ?? previousDefaultHeight) - previousDefaultHeight) < 0.01) {
        return normalizeAccessPoint({ ...ap, z: nextDefaultHeight });
      }
      return normalizeAccessPoint(ap);
    });
    document.getElementById('floorHeightText').innerText = `${state.floorHeight}m`;
    invalidateHeatmap();
    saveHistory();
    draw();
  }
});

document.getElementById('btnEditIconSize')?.addEventListener('click', async () => {
  const newVal = await customPrompt("Enter Icon Size multiplier (e.g., 1.0 for 100%, 0.5 for 50%):", state.iconSize);
  if (newVal && !isNaN(newVal)) {
    state.iconSize = parseFloat(newVal);
    document.getElementById('iconSizeText').innerText = `${Math.round(state.iconSize * 100)}%`;
    draw();
  }
});

document.getElementById('btnDeleteWallType')?.addEventListener('click', async () => {
  if (state.wallTypes.length <= 1) {
    await customAlert("Cannot delete the last wall type.");
    return;
  }
  if (await customConfirm(`Are you sure you want to delete this wall type?`)) {
    const index = state.wallTypes.findIndex(x => x.id === currentEditingId);
    if (index !== -1) {
      state.wallTypes.splice(index, 1);
      if (state.selectedWallType === currentEditingId) {
        state.selectedWallType = state.wallTypes[0].id;
      }
      renderWallPanel();
      renderEditWallList();
      selectWallToEdit(state.wallTypes[0].id);
    }
  }
});

document.getElementById('closeEditWallModal')?.addEventListener('click', () => {
  document.getElementById('editWallModal').style.display = 'none';
});

document.getElementById('closeWallPanel')?.addEventListener('click', () => {
  document.getElementById('drawWallsPanel').style.display = 'none';
  document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
  state.mode = null;
});

document.getElementById('btnApplyToAllWalls')?.addEventListener('click', async () => {
  const wallCount = state.elements.filter(e => e.type === 'wall').length;
  if (wallCount === 0) {
    await customAlert("No walls found to update.");
    return;
  }

  const selectedWall = state.wallTypes.find(w => w.id === state.selectedWallType);
  const confirmMsg = `Do you want to change all ${wallCount} walls to "${selectedWall.name}"?`;

  if (await customConfirm(confirmMsg, "Bulk Update Walls")) {
    state.elements = state.elements.map(e => {
      if (e.type === 'wall') {
        return {
          ...e,
          material: selectedWall.id,
          properties: selectedWall
        };
      }
      return e;
    });
    invalidateHeatmap();
    saveHistory();
    draw();
    await customAlert(`Successfully updated all walls to ${selectedWall.name}.`);
  }
});

document.getElementById('btnAutoTraceWalls')?.addEventListener('click', async () => {
  if (!state.blueprintImageSrc) {
    await customAlert("Please upload a floor plan image first.", "No Floor Plan");
    return;
  }
  if (!state.isScaleSet) {
    await customAlert("Please set the scale first so walls can be detected with correct dimensions.", "Scale Required");
    return;
  }

  // Show loading state
  const btn = document.getElementById('btnAutoTraceWalls');
  const originalText = btn.innerHTML;
  btn.innerHTML = '<span class="spinner"></span> Detecting...';
  btn.disabled = true;

  try {
    // Ask if user wants to clear existing walls first
    const existingWallCount = state.elements.filter(element => element.type === 'wall').length;
    let wallsCleared = false;
    if (existingWallCount > 0) {
      const clearFirst = await customConfirm(
        "Do you want to remove existing walls before auto-tracing? This prevents overlapping walls.",
        "Clear Existing Walls?"
      );
      if (clearFirst) {
        state.elements = state.elements.filter(element => element.type !== 'wall');
        wallsCleared = true;
      }
    }

    const resp = await fetch(`${API_BASE}/detect-walls`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: state.blueprintImageSrc,
        pixels_per_meter: state.pixelsPerMeter,
        min_wall_length_m: 0.8,
        wall_type: state.selectedWallType || 'thin',
        sensitivity: 55
      })
    });

    const data = await resp.json();

    if (data.success && data.walls && data.walls.length > 0) {
      // Add detected walls to state
      let addedCount = 0;
      let skippedCount = 0;
      const materialCounts = {};
      data.walls.forEach(wall => {
        const material = wall.wall_type || state.selectedWallType || 'thin';
        const detectedWall = {
          type: 'wall',
          points: wall.points,
          material
        };

        if (hasDuplicateWall(detectedWall, state.elements)) {
          skippedCount++;
          return;
        }

        state.elements.push(detectedWall);
        materialCounts[material] = (materialCounts[material] || 0) + 1;
        addedCount++;
      });

      if (addedCount > 0 || wallsCleared) {
        saveHistory();
        draw();
      }

      const skippedMessage = skippedCount > 0 ? `\nSkipped ${skippedCount} duplicate segments that were already on the plan.` : '';
      const filteredMessage = data.filtered_object_count > 0
        ? `\nIgnored ${data.filtered_object_count} likely furniture/object lines.`
        : '';
      const materialMessage = Object.entries(materialCounts)
        .map(([id, count]) => `${state.wallTypes.find(type => type.id === id)?.name || id}: ${count}`)
        .join(', ');
      await customAlert(`Successfully detected ${addedCount} new wall segments!${filteredMessage}${skippedMessage}\nMaterials: ${materialMessage}\n\nReview uncertain segments using the Select tool.`, "Auto Trace Complete");
    } else {
      if (wallsCleared) {
        saveHistory();
        draw();
      }
      await customAlert(data.message || "No walls were detected. Try increasing the sensitivity or uploading a clearer floor plan.", "No Walls Found");
    }
  } catch (err) {
    console.error("Auto trace error:", err);
    await customAlert("Failed to connect to the detection server. Make sure the backend is running.", "Connection Error");
  } finally {
    btn.innerHTML = originalText;
    btn.disabled = false;
    // Re-render lucide icons inside the button
    if (window.lucide) lucide.createIcons();
  }
});

document.getElementById('btnAutoDetectRooms')?.addEventListener('click', async () => {
  if (state.elements.filter(element => element.type === 'wall').length < 3) {
    await customAlert("Draw or auto-trace walls first, then run Auto Rooms.", "Not Enough Walls");
    return;
  }

  const exteriorWallsAdded = ensureExteriorBoundaryWalls();
  let rooms = [];
  if (blueprintHasFilledWallStyle()) {
    rooms = await detectFullRoomsFromBlueprint();
    if (rooms.length > 0) {
      rooms = augmentFullRoomsWithUncoveredRectangles(rooms);
    }
  }
  if (rooms.length === 0) {
    rooms = detectRoomsFromWalls();
  }
  if (rooms.length === 0) {
    state.rooms = [];
    state.showRoomsOverlay = false;
    state.autoPlacementLocked = false;
    renderRoomsPanel();
    renderAutoPlacementPanel();
    draw();
    await customAlert("No closed rooms were detected. Try closing large wall gaps or drawing the outer boundary first.", "No Rooms Found");
    return;
  }

  state.rooms = rooms;
  state.serviceBoundary = generateServiceBoundaryFromWalls();
  state.doorOpenings = detectDoorOpenings(getWallSegments());
  state.showRoomsOverlay = true;
  state.autoPlacementLocked = true;
  if (state.apPlacementTab !== 'auto') {
    switchApPlacementTab('auto');
  }
  setHeatmapEnabled(false, false);
  syncApPlacementTabs();
  renderRoomsPanel();
  renderAutoPlacementPanel();
  saveHistory();
  draw();
  const stairCount = rooms.filter(room => room.roomType === 'stairs').length;
  const stairMessage = stairCount > 0
    ? `\nDetected ${stairCount} stair area${stairCount === 1 ? '' : 's'} and excluded them from AP placement.`
    : '';
  const openingMessage = state.doorOpenings.length > 0
    ? `\nRecognized ${state.doorOpenings.length} door/opening gaps for room closure and signal propagation.`
    : '';
  const exteriorMessage = exteriorWallsAdded > 0
    ? `\nClosed ${exteriorWallsAdded} missing exterior wall segment${exteriorWallsAdded === 1 ? '' : 's'} from the blueprint footprint.`
    : '';
  await customAlert(`Detected ${rooms.length} rooms/areas.${exteriorMessage}${stairMessage}${openingMessage}\nOpen-plan spaces without a physical separating wall are kept as one RF service area.\nThese zones can be used later for RL demand and placement rewards.`, "Auto Rooms Complete");
});

async function detectFullRoomsFromBlueprint() {
  if (!state.blueprintImageSrc || !state.isScaleSet) return [];
  try {
    const response = await fetch(`${API_BASE}/detect-rooms`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: state.blueprintImageSrc,
        pixels_per_meter: state.pixelsPerMeter,
        min_room_area_m2: 1.2,
        max_room_area_m2: 180
      })
    });
    const data = await response.json();
    if (!data.success || !Array.isArray(data.rooms)) return [];

    const walls = getWallSegments();
    const rooms = data.rooms.map((room, index) => ({
      id: `room_image_${Date.now()}_${index + 1}`,
      name: `Room ${index + 1}`,
      type: 'room',
      roomType: 'office',
      areaM2: Number(room.areaM2 || polygonArea(room.polygon || [])),
      centroid: room.centroid || polygonCentroid(room.polygon || []),
      polygon: room.polygon || [],
      userDensity: 0.1,
      clients: estimateRoomClients(Number(room.areaM2 || polygonArea(room.polygon || []))),
      priority: estimateRoomPriority(Number(room.areaM2 || polygonArea(room.polygon || []))),
      coverageTarget: -67,
      excluded: false
    })).filter(room => room.polygon.length >= 3);
    return normalizeRooms(classifyDetectedRoomTypes(rooms, walls));
  } catch (error) {
    console.warn('Filled-room detection failed; using wall fallback.', error);
    return [];
  }
}

function augmentFullRoomsWithUncoveredRectangles(fullRooms) {
  const walls = getWallSegments();
  const fallbackRooms = detectRectangularRoomsFromWalls(walls, []);
  const merged = [...fullRooms];

  fallbackRooms
    .sort((a, b) => Number(b.areaM2 || 0) - Number(a.areaM2 || 0))
    .forEach(room => {
      const overlapsFullRoom = merged.some(existing =>
        polygonOverlapRatioApprox(room.polygon, existing.polygon) > 0.18
        || polygonOverlapRatioApprox(existing.polygon, room.polygon) > 0.18
      );
      if (!overlapsFullRoom) merged.push(room);
    });

  const sorted = merged
    .sort((a, b) => a.centroid.y - b.centroid.y || a.centroid.x - b.centroid.x)
    .map((room, index) => ({ ...room, name: `Room ${index + 1}` }));
  return normalizeRooms(classifyDetectedRoomTypes(sorted, walls));
}

document.getElementById('btnClearRooms')?.addEventListener('click', () => {
  state.rooms = [];
  state.serviceBoundary = [];
  state.selectedRoomIndex = null;
  state.autoPlacementLocked = false;
  renderRoomsPanel();
  syncApPlacementTabs();
  renderAutoPlacementPanel();
  saveHistory();
  draw();
});

document.getElementById('btnToggleRoomsOverlay')?.addEventListener('click', () => {
  state.showRoomsOverlay = !state.showRoomsOverlay;
  renderRoomsPanel();
  draw();
});

document.getElementById('btnRunRlPpoPlacement')?.addEventListener('click', runRlPpoPlacement);
document.getElementById('btnRunRlTrainingBatch')?.addEventListener('click', runRlTrainingBatch);

// Initialize UI
setTimeout(() => {
  const selectBtn = document.getElementById('btnSelect');
  if (selectBtn) setActiveTool('select', selectBtn);
}, 200);

function renderWallPanel() {
  const container = document.getElementById('wallListContainer');
  if (!container) return;
  container.innerHTML = '';
  const use5GHz = getActiveFrequencyGHz() >= 5;

  state.wallTypes.forEach(w => {
    const item = document.createElement('div');
    item.className = `wall-item ${state.selectedWallType === w.id ? 'active' : ''}`;
    item.onclick = () => {
      if (state.selectedElementType === 'wall' && state.selectedElementIndex !== null) {
        state.elements[state.selectedElementIndex].material = w.id;
        saveHistory();
        draw();
      } else {
        state.selectedWallType = w.id;
      }
      renderWallPanel();
    };

    item.innerHTML = `
      <div class="wall-preview" style="background-color: ${w.color};"></div>
      <span class="wall-name">${w.name}</span>
      <span class="wall-db">${use5GHz ? w.db5 : w.db24}dB</span>
    `;
    container.appendChild(item);
  });
}

document.getElementById('btnEditWallTypes')?.addEventListener('click', () => {
  openEditWallModal();
});

function renderDevicePanel() {
  const container = document.getElementById('deviceListContainer');
  if (!container) return;
  container.innerHTML = '';
  const activeFrequency = getActiveFrequencyGHz();

  state.apModels.forEach(ap => {
    const activeGain = getAntennaGainForBand(ap, ap, activeFrequency);
    const activeDefaultPower = getDefaultTxPowerForBand(ap, activeFrequency);
    const item = document.createElement('div');
    item.className = `device-item ${state.selectedApModel === ap.id ? 'active' : ''}`;
    item.onclick = () => {
      state.selectedApModel = ap.id;
      renderDevicePanel();
    };

    item.innerHTML = `
      <div class="device-image-preview">
        <img src="${ap.image}" onerror="this.src='https://via.placeholder.com/40?text=AP'" alt="${ap.model}">
      </div>
      <div class="device-info">
        <span class="device-name">${ap.model}</span>
        <span class="device-vendor">${ap.vendor}</span>
      </div>
      <div class="device-actions">
        <span class="device-gain">${activeGain}dBi / ${activeDefaultPower}dBm</span>
        <button class="detail-btn" onclick="event.stopPropagation(); showDeviceDetails('${ap.id}')">Detail</button>
      </div>
    `;
    container.appendChild(item);
  });
}

function switchApPlacementTab(nextTab) {
  const normalizedTab = nextTab === 'auto' ? 'auto' : 'manual';
  if (state.apPlacementTab === normalizedTab) {
    syncApPlacementTabs();
    return;
  }

  if (state.apPlacementTab === 'manual') {
    state.manualAccessPointsBackup = JSON.parse(JSON.stringify(state.accessPoints || []));
    state.accessPoints = normalizeAccessPoints(JSON.parse(JSON.stringify(state.autoAccessPointsBackup || [])));
  } else {
    state.autoAccessPointsBackup = JSON.parse(JSON.stringify(state.accessPoints || []));
    state.accessPoints = normalizeAccessPoints(JSON.parse(JSON.stringify(state.manualAccessPointsBackup || [])));
  }

  state.apPlacementTab = normalizedTab;
  if (normalizedTab === 'manual') {
    state.showRoomsOverlay = false;
    setHeatmapEnabled(true);
  } else {
    state.showRoomsOverlay = (state.rooms || []).length > 0;
    setHeatmapEnabled(false, false);
  }
  if (normalizedTab !== 'manual') invalidateHeatmap();
  syncApPlacementTabs();
  renderRoomsPanel();
  renderDevicePanel();
  renderDeviceList();
  draw();
}

function syncApPlacementTabs() {
  document.querySelectorAll('.ap-placement-tabs button').forEach(button => {
    const isActive = button.dataset.apTab === state.apPlacementTab;
    button.classList.toggle('active', isActive);
    button.style.display = '';
  });

  const tabs = document.querySelector('.ap-placement-tabs');
  if (tabs) tabs.classList.remove('auto-only');

  const manualPanel = document.getElementById('manualPlacementTab');
  const autoPanel = document.getElementById('autoPlacementTab');
  const footer = document.getElementById('devicePanelFooter');
  if (manualPanel) manualPanel.style.display = state.apPlacementTab === 'manual' ? 'block' : 'none';
  if (autoPanel) autoPanel.style.display = state.apPlacementTab === 'auto' ? 'block' : 'none';
  if (footer) footer.style.display = state.apPlacementTab === 'manual' ? 'flex' : 'none';

  if (state.mode === 'ap') {
    canvas.style.cursor = state.apPlacementTab === 'auto' ? 'default' : 'crosshair';
  }
}

document.querySelectorAll('.ap-placement-tabs button').forEach(button => {
  button.addEventListener('click', () => {
    switchApPlacementTab(button.dataset.apTab || 'manual');
    renderAutoPlacementPanel();
  });
});

function renderAutoPlacementPanel() {
  const rooms = state.rooms || [];
  const count = document.getElementById('autoRoomsCount');
  const status = document.getElementById('autoPlacementStatus');
  const list = document.getElementById('autoRoomsList');
  if (count) count.textContent = `${rooms.length} detected`;
  if (status) status.textContent = rooms.length > 0 ? 'Ready' : 'Rooms required';
  if (!list) return;

  list.innerHTML = '';
  if (rooms.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'auto-room-item';
    empty.style.gridTemplateColumns = '1fr';
    empty.textContent = 'Detect room areas before running automatic placement.';
    list.appendChild(empty);
    return;
  }

  rooms.forEach((room, index) => {
    const item = document.createElement('div');
    item.className = `auto-room-item ${state.selectedRoomIndex === index ? 'active' : ''} ${room.excluded ? 'excluded' : ''}`;
    item.innerHTML = `
      <span class="room-swatch"></span>
      <span>${room.name}${room.roomType === 'stairs' ? ' (No AP)' : ''}</span>
      <span class="room-area">${Number(room.areaM2 || 0).toFixed(1)}m²</span>
    `;
    item.addEventListener('click', () => openEditRoomModal(index));
    list.appendChild(item);
  });
}

function createAccessPointAt(x, y, name) {
  const model = getApModelById(state.selectedApModel);
  const power24 = getDefaultTxPowerForBand(model, 2.4);
  const power5 = getDefaultTxPowerForBand(model, 5);
  return normalizeAccessPoint({
    name,
    model_id: model.id,
    model: model.model,
    vendor: model.vendor,
    x,
    y,
    z: getDefaultApHeightM(),
    power: getActiveFrequencyGHz() >= 5 ? power5 : power24,
    power24,
    power5,
    gain24: model.gain24,
    gain5: model.gain5,
    frequency: getActiveFrequencyGHz()
  });
}

let smartThinkingStageTimer = null;
let smartThinkingProgressTimer = null;
let alternativePlansSession = null;

function startSmartPlacementThinking(roomCount) {
  const overlay = document.getElementById('smartPlacementThinking');
  const status = document.getElementById('smartThinkingStatus');
  const detail = document.getElementById('smartThinkingDetail');
  const progressBar = document.getElementById('smartThinkingProgress');
  const panelStatus = document.getElementById('autoPlacementStatus');
  if (!overlay || !status || !detail || !progressBar) return;

  const stages = [
    ['Reading rooms and service areas...', `${roomCount} active rooms found`],
    ['Generating valid AP candidate positions...', 'Avoiding walls and duplicate room placements'],
    ['Exploring several placement alternatives...', 'Running stable and exploratory agent rollouts'],
    ['Testing signal coverage through walls...', 'Comparing RSSI across every alternative'],
    ['Evaluating capacity and interference...', 'Checking SINR and channel overlap for each plan'],
    ['Removing costly or redundant APs...', 'Keeping only placements with useful coverage'],
    ['Selecting the best alternative...', 'Balancing coverage, capacity, interference, and AP cost']
  ];
  let stageIndex = 0;
  let progress = 5;

  status.textContent = stages[0][0];
  detail.textContent = stages[0][1];
  progressBar.style.width = `${progress}%`;
  overlay.classList.add('active');
  overlay.setAttribute('aria-hidden', 'false');
  if (panelStatus) panelStatus.textContent = 'Agent thinking...';
  if (window.lucide) lucide.createIcons();

  clearInterval(smartThinkingStageTimer);
  clearInterval(smartThinkingProgressTimer);
  smartThinkingStageTimer = setInterval(() => {
    stageIndex = Math.min(stageIndex + 1, stages.length - 1);
    status.textContent = stages[stageIndex][0];
    detail.textContent = stages[stageIndex][1];
  }, 1450);
  smartThinkingProgressTimer = setInterval(() => {
    progress = Math.min(92, progress + Math.max(1, Math.round((92 - progress) * 0.08)));
    progressBar.style.width = `${progress}%`;
  }, 280);
}

async function stopSmartPlacementThinking(success = true) {
  const overlay = document.getElementById('smartPlacementThinking');
  const status = document.getElementById('smartThinkingStatus');
  const detail = document.getElementById('smartThinkingDetail');
  const progressBar = document.getElementById('smartThinkingProgress');
  clearInterval(smartThinkingStageTimer);
  clearInterval(smartThinkingProgressTimer);
  smartThinkingStageTimer = null;
  smartThinkingProgressTimer = null;
  if (!overlay) return;

  if (status) status.textContent = success ? 'Placement plan ready' : 'Placement analysis stopped';
  if (detail) detail.textContent = success ? 'Applying optimized AP locations to the floor plan' : 'The agent could not complete this placement';
  if (progressBar) progressBar.style.width = '100%';
  await new Promise(resolve => setTimeout(resolve, success ? 500 : 250));
  overlay.classList.remove('active');
  overlay.setAttribute('aria-hidden', 'true');
}

function previewAlternativePlan(index) {
  const plan = alternativePlansSession?.plans.find(item => Number(item.index) === Number(index));
  if (!plan) return;
  alternativePlansSession.previewIndex = Number(plan.index);
  state.accessPoints = normalizeAccessPoints(plan.access_points || []);
  state.autoAccessPointsBackup = JSON.parse(JSON.stringify(state.accessPoints));
  invalidateHeatmap();
  setHeatmapEnabled(true, false);
  renderDevicePanel();
  renderDeviceList();
  draw();
  renderAlternativePlansModal();
}

function renderAlternativePlansModal() {
  const session = alternativePlansSession;
  const list = document.getElementById('alternativePlansList');
  const details = document.getElementById('alternativePlanDetails');
  const status = document.getElementById('alternativePreviewStatus');
  if (!session || !list || !details) return;

  list.innerHTML = '';
  session.plans.forEach(plan => {
    const metrics = plan.metrics || {};
    const card = document.createElement('div');
    card.className = `alternative-plan-card ${Number(plan.index) === session.previewIndex ? 'active' : ''} ${Number(plan.index) === session.bestIndex ? 'best' : ''}`;
    card.innerHTML = `
      <div class="alternative-plan-card-header">
        <span class="alternative-plan-name">Plan ${plan.index}</span>
        <span class="alternative-plan-mode">${plan.mode === 'stable' ? 'Stable' : 'Exploration'}</span>
      </div>
      <div class="alternative-plan-card-metrics">
        <div class="alternative-plan-metric"><span>AP</span><strong>${(plan.access_points || []).length}</strong></div>
        <div class="alternative-plan-metric"><span>Coverage</span><strong>${(Number(metrics.sample_coverage_ratio || 0) * 100).toFixed(0)}%</strong></div>
        <div class="alternative-plan-metric"><span>Floor</span><strong>${(Number(metrics.floor_coverage_ratio || 0) * 100).toFixed(0)}%</strong></div>
        <div class="alternative-plan-metric"><span>Interference</span><strong>${(Number(metrics.cochannel_interference_ratio || 0) * 100).toFixed(1)}%</strong></div>
      </div>
    `;
    card.addEventListener('click', () => previewAlternativePlan(plan.index));
    list.appendChild(card);
  });

  const plan = session.plans.find(item => Number(item.index) === session.previewIndex);
  const metrics = plan?.metrics || {};
  details.innerHTML = `
    <h4>Plan ${plan?.index || '-'} details</h4>
    <div class="alternative-detail-row"><span>Agent mode</span><strong>${plan?.mode === 'stable' ? 'Stable' : 'Exploration'}</strong></div>
    <div class="alternative-detail-row"><span>Placement score</span><strong>${Number(plan?.score || 0).toFixed(2)}</strong></div>
    <div class="alternative-detail-row"><span>Access points</span><strong>${(plan?.access_points || []).length}</strong></div>
    <div class="alternative-detail-row"><span>Room sample coverage</span><strong>${(Number(metrics.sample_coverage_ratio || 0) * 100).toFixed(1)}%</strong></div>
    <div class="alternative-detail-row"><span>Whole-building coverage</span><strong>${(Number(metrics.floor_coverage_ratio || 0) * 100).toFixed(1)}%</strong></div>
    <div class="alternative-detail-row"><span>Capacity</span><strong>${(Number(metrics.capacity_ratio || 0) * 100).toFixed(1)}%</strong></div>
    <div class="alternative-detail-row"><span>Weak rooms</span><strong>${Number(metrics.weak_room_count || 0).toFixed(0)}</strong></div>
    <div class="alternative-detail-row"><span>Co-channel interference</span><strong>${(Number(metrics.cochannel_interference_ratio || 0) * 100).toFixed(1)}%</strong></div>
  `;
  if (status) {
    status.textContent = Number(plan?.index) === session.bestIndex
      ? `Previewing Plan ${plan?.index}, selected as the best score`
      : `Previewing Plan ${plan?.index}; apply it to replace the recommended plan`;
  }
}

function openAlternativePlansModal(data) {
  const plans = Array.isArray(data.alternative_plans) ? data.alternative_plans : [];
  if (plans.length < 2) return false;
  alternativePlansSession = {
    plans,
    bestIndex: Number(data.selected_alternative || plans[0].index),
    previewIndex: Number(data.selected_alternative || plans[0].index)
  };
  document.getElementById('alternativePlansModal').style.display = 'flex';
  renderAlternativePlansModal();
  if (window.lucide) lucide.createIcons();
  return true;
}

function closeAlternativePlansModal(applyPreview = false) {
  const session = alternativePlansSession;
  if (!session) return;
  const targetIndex = applyPreview ? session.previewIndex : session.bestIndex;
  const plan = session.plans.find(item => Number(item.index) === targetIndex);
  if (plan) {
    state.accessPoints = normalizeAccessPoints(plan.access_points || []);
    state.autoAccessPointsBackup = JSON.parse(JSON.stringify(state.accessPoints));
    invalidateHeatmap();
    saveHistory();
    renderDevicePanel();
    renderDeviceList();
    draw();
  }
  document.getElementById('alternativePlansModal').style.display = 'none';
  alternativePlansSession = null;
}

document.getElementById('closeAlternativePlansModal')?.addEventListener('click', () => closeAlternativePlansModal(false));
document.getElementById('btnCancelAlternativePlan')?.addEventListener('click', () => closeAlternativePlansModal(false));
document.getElementById('btnApplyAlternativePlan')?.addEventListener('click', () => closeAlternativePlansModal(true));

async function runRlPpoPlacement() {
  const rooms = (state.rooms || []).filter(room => room.centroid && !room.excluded);
  if (rooms.length === 0) {
    await customAlert("Run Auto Detect Rooms first. Smart AP placement needs room/area data.", "Rooms Required");
    return;
  }

  if (state.accessPoints.length > 0) {
    const replace = await customConfirm("Replace existing APs with Smart AP placement?", "Smart AP Placement");
    if (!replace) return;
    state.accessPoints = [];
  }

  const btn = document.getElementById('btnRunRlPpoPlacement');
  const originalText = btn?.innerHTML;
  if (btn) {
    btn.innerHTML = '<span class="spinner"></span> Running Smart AI...';
    btn.disabled = true;
  }
  startSmartPlacementThinking(rooms.length);

  try {
    const projectData = buildProjectPayload();

    const resp = await fetch(`${API_BASE}/rl/ppo-placement`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(projectData)
    });

    if (!resp.ok) {
      let detail = `Smart AP placement failed with ${resp.status}`;
      try {
        const errorData = await resp.json();
        detail = errorData.detail || detail;
      } catch (_) {
        // Keep the status-based fallback when the backend did not return JSON.
      }
      throw new Error(detail);
    }
    const data = await resp.json();
    const aps = Array.isArray(data.access_points) ? data.access_points : [];

    if (aps.length === 0) {
      await stopSmartPlacementThinking(false);
      await customAlert("Smart AP did not return any AP placement.", "No APs Placed");
      return;
    }

    state.accessPoints = normalizeAccessPoints(aps);
    state.autoAccessPointsBackup = JSON.parse(JSON.stringify(state.accessPoints));
    setHeatmapEnabled(true);
    saveHistory();
    renderDevicePanel();
    renderDeviceList();
    draw();

    const metrics = data.metrics || {};
    const coverage = Number(metrics.coverage_ratio ?? 0) * 100;
    const sampleCoverage = Number(metrics.sample_coverage_ratio ?? metrics.coverage_ratio ?? 0) * 100;
    const capacity = Number(metrics.capacity_ratio ?? 0) * 100;
    const capacityShortfallRooms = Number(metrics.capacity_shortfall_room_count ?? 0);
    const denseRoomsWithoutLocalAp = Number(metrics.high_density_room_without_local_ap_count ?? 0);
    const blankspot = Number(metrics.blankspot_ratio ?? Math.max(0, 1 - (metrics.sample_coverage_ratio ?? metrics.coverage_ratio ?? 0))) * 100;
    const overlap = Number(metrics.overlap_ratio ?? 0) * 100;
    const weakRooms = Number(metrics.weak_room_count ?? 0);
    const floorCoverage = Number(metrics.floor_coverage_ratio ?? metrics.sample_coverage_ratio ?? 0) * 100;
    const floorSinrCoverage = Number(metrics.floor_sinr_coverage_ratio ?? metrics.sinr_coverage_ratio ?? 0) * 100;
    const interference = Number(metrics.cochannel_interference_ratio ?? 0) * 100;
    const channelSummary = data.channel_summary || {};
    const channelConflicts = Number(channelSummary.cochannel_conflicts ?? 0);
    const channelBand = channelSummary.active_band || getActiveFrequencyLabel();
    const ppoApCount = Number(data.ppo_ap_count ?? aps.length);
    const prunedApCount = Number(data.pruned_ap_count ?? 0);
    const supplementalApCount = Number(data.supplemental_ap_count ?? 0);
    const serviceRoomCount = Number(data.service_room_count ?? 0);
    const noServiceRoomCount = Number(data.no_service_room_count ?? 0);
    const alternativesEvaluated = Number(data.alternatives_evaluated ?? 1);
    const selectedAlternative = Number(data.selected_alternative ?? 1);
    const selectedAlternativeMode = data.selected_alternative_mode === 'stable' ? 'stable' : 'exploration';
    const requirementStatus = data.requirements_met ? 'All coverage requirements met' : 'Coverage requirements not fully met';
    const economics = data.economics || {};
    const rfSummary = data.rf_summary || {};
    const estimatedCost = Number(economics.estimated_hardware_cost ?? 0);
    const costText = estimatedCost > 0
      ? `${economics.vendor || ''} ${economics.model || ''}, estimated hardware cost: $${estimatedCost.toFixed(0)}`
      : 'Hardware cost estimate unavailable';
    const wallMaterialText = (data.wall_materials || [])
      .map(material => {
        const attenuation = material.attenuation_db || {};
        return `${material.material_name}: ${material.count} walls, ${Number(attenuation['2.4'] ?? 0).toFixed(0)} dB @2.4 GHz / ${Number(attenuation['5'] ?? 0).toFixed(0)} dB @5 GHz`;
      })
      .join('; ');
    const alternativeSummary = (data.alternative_summaries || [])
      .map(alternative => {
        const planCoverage = Number(alternative.sample_coverage_ratio ?? 0) * 100;
        const planInterference = Number(alternative.cochannel_interference_ratio ?? 0) * 100;
        const selectedMark = Number(alternative.index) === selectedAlternative ? ' [selected]' : '';
        return `Plan ${alternative.index}: ${alternative.ap_count} AP, ${planCoverage.toFixed(0)}% coverage, ${planInterference.toFixed(1)}% interference${selectedMark}`;
      })
      .join('\n');
    const alternativeText = alternativeSummary ? `\n\nAlternatives compared:\n${alternativeSummary}` : '';
    await stopSmartPlacementThinking(true);
    if (btn) {
      btn.innerHTML = originalText;
      btn.disabled = false;
      if (window.lucide) lucide.createIcons();
    }
    await customAlert(
      `${requirementStatus}\nThe agent automatically selected Plan ${selectedAlternative} (${selectedAlternativeMode}) as the best of ${alternativesEvaluated} evaluated plans.\nPlaced ${aps.length} APs using the latest Smart AP model.\n${costText}\nRF simulation: ${Number(rfSummary.frequency_ghz ?? getActiveFrequencyGHz()).toFixed(1)} GHz, ${Number(rfSummary.tx_power_dbm ?? 0).toFixed(0)} dBm TX, good signal >= ${Number(rfSummary.good_signal_dbm ?? -67).toFixed(0)} dBm\nPlanning capacity: ${Number(rfSummary.planning_clients_per_ap ?? 0).toFixed(0)} active clients per AP\nService rooms: ${serviceRoomCount}, empty/no-service rooms: ${noServiceRoomCount}\nWall materials used: ${wallMaterialText || 'none'}\nPPO selected: ${ppoApCount}, redundant removed: ${prunedApCount}, cost-aware additions: ${supplementalApCount}\nRoom coverage: ${coverage.toFixed(0)}%\nDetected-room samples: ${sampleCoverage.toFixed(0)}%\nWhole-building coverage: ${floorCoverage.toFixed(0)}%\nWhole-building SINR coverage: ${floorSinrCoverage.toFixed(0)}%\nBlank spot: ${blankspot.toFixed(0)}%\nCapacity: ${capacity.toFixed(0)}%\nRooms below capacity: ${capacityShortfallRooms.toFixed(0)}\nDense rooms without local AP: ${denseRoomsWithoutLocalAp.toFixed(0)}\nOverlap: ${overlap.toFixed(0)}%\nCo-channel interference: ${interference.toFixed(1)}%\nWeak rooms: ${weakRooms.toFixed(0)}\nChannel plan: ${channelBand}, ${channelConflicts} nearby same-channel pairs\nCandidates evaluated per plan: ${data.candidate_count || 0}${alternativeText}`,
      "Best Smart AP Plan Applied"
    );
  } catch (err) {
    console.error("Smart AP placement error:", err);
    await stopSmartPlacementThinking(false);
    await customAlert(`Failed to run Smart AP placement.\n${err.message || err}`, "Smart AP Error");
  } finally {
    const overlay = document.getElementById('smartPlacementThinking');
    if (overlay?.classList.contains('active')) {
      await stopSmartPlacementThinking(false);
    }
    if (btn) {
      btn.innerHTML = originalText;
      btn.disabled = false;
      if (window.lucide) lucide.createIcons();
    }
    renderAutoPlacementPanel();
  }
}

async function runRlTrainingBatch() {
  const rooms = (state.rooms || []).filter(room => room.centroid && !room.excluded);
  if (rooms.length === 0) {
    await customAlert("Run Auto Detect Rooms first. Training variants need room/area data.", "Rooms Required");
    return;
  }

  const variantsInput = await customPrompt(
    "How many wall/demand variants should be generated? Start small for UI runs.",
    "20",
    "Training Batch Variants"
  );
  if (variantsInput === null) return;

  const variants = clamp(parseInt(variantsInput, 10) || 20, 1, 200);
  const episodesInput = await customPrompt(
    "How many random baseline episodes per variant?",
    "10",
    "Training Batch Episodes"
  );
  if (episodesInput === null) return;

  const episodes = clamp(parseInt(episodesInput, 10) || 10, 0, 500);
  const proceed = await customConfirm(
    `Generate ${variants} wall/demand variants from this floor plan?\nThis may take a while and results will be saved in training_runs/.`,
    "Generate Training Batch"
  );
  if (!proceed) return;

  const btn = document.getElementById('btnRunRlTrainingBatch');
  const originalText = btn?.innerHTML;
  if (btn) {
    btn.innerHTML = '<span class="spinner"></span> Training...';
    btn.disabled = true;
  }

  try {
    const projectData = buildProjectPayload();
    projectData.parameters.rlTrainingVariants = variants;
    projectData.parameters.rlTrainingEpisodes = episodes;
    projectData.parameters.rlTrainingSeed = 42;
    projectData.parameters.rlGreedyCandidateLimit = 80;
    projectData.parameters.rlSaveTrainingScenarios = false;

    const resp = await fetch(`${API_BASE}/rl/self-training`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(projectData)
    });

    if (!resp.ok) throw new Error(`Training batch failed with ${resp.status}`);
    const data = await resp.json();
    const summary = data.summary || {};
    const averages = summary.averages || {};
    const outputs = summary.outputs || {};
    const avgCoverage = Number(averages.sample_coverage_ratio ?? 0) * 100;
    const avgCapacity = Number(averages.capacity_ratio ?? 0) * 100;
    const avgApCount = Number(averages.ap_count ?? 0);
    const avgWeakRooms = Number(averages.weak_room_count ?? 0);
    if (btn) {
      btn.innerHTML = originalText;
      btn.disabled = false;
      if (window.lucide) lucide.createIcons();
    }

    await customAlert(
      `Generated ${summary.variant_count || variants} training variants.\nAverage area samples: ${avgCoverage.toFixed(0)}%\nAverage capacity: ${avgCapacity.toFixed(0)}%\nAverage AP count: ${avgApCount.toFixed(1)}\nAverage weak rooms: ${avgWeakRooms.toFixed(1)}\nSaved to:\n${outputs.directory || 'training_runs/'}`,
      "Training Batch Complete"
    );
  } catch (err) {
    console.error("RL training batch error:", err);
    await customAlert("Failed to generate training batch. Make sure the backend is running.", "Training Batch Error");
  } finally {
    if (btn) {
      btn.innerHTML = originalText;
      btn.disabled = false;
      if (window.lucide) lucide.createIcons();
    }
  }
}

window.showDeviceDetails = (id) => {
  const ap = getApModelById(id);
  if (!ap) return;

  const modal = document.getElementById('deviceDetailsModal');
  const content = document.getElementById('deviceDetailsContent');
  const activeFrequency = getActiveFrequencyGHz();
  const simulationProfile = getSimulationProfile(activeFrequency);

  content.innerHTML = `
    <div class="details-grid">
      <div class="details-image">
        <img src="${ap.image}" onerror="this.src='https://via.placeholder.com/200?text=AP'" alt="${ap.model}">
      </div>
      <div class="details-specs">
        <h2>${ap.vendor} ${ap.model}</h2>
        <div class="spec-section">
          <h4>Wireless Features</h4>
          <div class="spec-row"><span>Launched</span><strong>${ap.launched || 'N/A'}</strong></div>
          <div class="spec-row"><span>Coverage</span><strong>${ap.coverage || 'N/A'}</strong></div>
          <div class="spec-row"><span>MIMO</span><strong>${ap.mimo}</strong></div>
          <div class="spec-row"><span>Signal Rate (5GHz)</span><strong>${ap.rate5} Mbps</strong></div>
          <div class="spec-row"><span>Signal Rate (2.4GHz)</span><strong>${ap.rate24} Mbps</strong></div>
          <div class="spec-row"><span>Antenna Gain (2.4GHz)</span><strong>${ap.gain24} dBi</strong></div>
          <div class="spec-row"><span>Antenna Gain (5GHz)</span><strong>${ap.gain5} dBi</strong></div>
          <div class="spec-row"><span>Max TX Power</span><strong>${ap.maxPower} dBm</strong></div>
          <div class="spec-row"><span>Default Sim TX (2.4GHz)</span><strong>${getDefaultTxPowerForBand(ap, 2.4)} dBm</strong></div>
          <div class="spec-row"><span>Default Sim TX (5GHz)</span><strong>${getDefaultTxPowerForBand(ap, 5)} dBm</strong></div>
          <div class="spec-row"><span>Concurrent Clients</span><strong>${ap.clients}+</strong></div>
          <div class="spec-row"><span>BSSIDs</span><strong>${ap.bssids || '8 per Radio'}</strong></div>
          <div class="spec-row"><span>Standards</span><strong>${ap.standards || 'Wi-Fi 6'}</strong></div>
        </div>
        <div class="spec-section" style="margin-top: 16px;">
          <h4>Simulation Profile</h4>
          <p style="font-size: 0.8rem; color: #64748b; line-height: 1.4;">
            Active band: ${getActiveFrequencyLabel()}.
            Path-loss exponent ${simulationProfile.pathLossExponent},
            fade margin ${simulationProfile.fadeMarginDb} dB,
            wall multiplier ${simulationProfile.wallLossMultiplier.toFixed(2)}x.
          </p>
        </div>
      </div>
    </div>
  `;

  modal.style.display = 'flex';
};

document.getElementById('closeDeviceDetails')?.addEventListener('click', () => {
  document.getElementById('deviceDetailsModal').style.display = 'none';
});

let currentEditingDeviceId = null;

function openEditDeviceModal() {
  const modal = document.getElementById('editDeviceModal');
  modal.style.display = 'flex';
  renderEditDeviceList();
  selectDeviceToEdit(state.selectedApModel || state.apModels[0]?.id);
}

function renderEditDeviceList() {
  const list = document.getElementById('editDeviceList');
  if (!list) return;
  list.innerHTML = '';

  state.apModels.forEach(apModel => {
    const item = document.createElement('div');
    item.className = 'edit-wall-item';
    item.dataset.id = apModel.id;
    item.innerHTML = `
      <div class="wall-preview" style="background-color: ${apModel.color};"></div>
      <span>${apModel.vendor} ${apModel.model}</span>
      <span class="db-val">${getDefaultTxPowerForBand(apModel, 5)}dBm</span>
    `;
    item.onclick = () => selectDeviceToEdit(apModel.id);
    list.appendChild(item);
  });
}

function selectDeviceToEdit(id) {
  currentEditingDeviceId = id;
  const apModel = getApModelById(id);
  if (!apModel) return;

  document.querySelectorAll('#editDeviceList .edit-wall-item').forEach(element => {
    element.classList.toggle('active', element.dataset.id === id);
  });

  document.getElementById('editDeviceVendor').value = apModel.vendor;
  document.getElementById('editDeviceModel').value = apModel.model;
  document.getElementById('editDeviceGain24').value = apModel.gain24;
  document.getElementById('editDeviceGain5').value = apModel.gain5;
  document.getElementById('editDeviceMaxPower').value = apModel.maxPower;
  document.getElementById('editDeviceDefaultPower24').value = apModel.defaultPower24;
  document.getElementById('editDeviceDefaultPower5').value = apModel.defaultPower5;
  document.getElementById('editDeviceRate24').value = apModel.rate24;
  document.getElementById('editDeviceRate5').value = apModel.rate5;
  document.getElementById('editDeviceClients').value = apModel.clients;
  document.getElementById('editDeviceStandards').value = apModel.standards;
}

document.getElementById('btnEditDeviceTypes')?.addEventListener('click', () => {
  openEditDeviceModal();
});

document.getElementById('closeEditDeviceModal')?.addEventListener('click', () => {
  document.getElementById('editDeviceModal').style.display = 'none';
});

document.getElementById('btnSaveDeviceType')?.addEventListener('click', async () => {
  const apModel = state.apModels.find(model => model.id === currentEditingDeviceId);
  if (!apModel) return;

  apModel.vendor = document.getElementById('editDeviceVendor').value.trim() || apModel.vendor;
  apModel.model = document.getElementById('editDeviceModel').value.trim() || apModel.model;
  apModel.gain24 = Number(document.getElementById('editDeviceGain24').value);
  apModel.gain5 = Number(document.getElementById('editDeviceGain5').value);
  apModel.maxPower = Number(document.getElementById('editDeviceMaxPower').value);
  apModel.defaultPower24 = clamp(Number(document.getElementById('editDeviceDefaultPower24').value), 1, apModel.maxPower);
  apModel.defaultPower5 = clamp(Number(document.getElementById('editDeviceDefaultPower5').value), 1, apModel.maxPower);
  apModel.rate24 = Number(document.getElementById('editDeviceRate24').value);
  apModel.rate5 = Number(document.getElementById('editDeviceRate5').value);
  apModel.clients = Number(document.getElementById('editDeviceClients').value);
  apModel.standards = document.getElementById('editDeviceStandards').value.trim() || apModel.standards;
  Object.assign(apModel, normalizeApModel(apModel));
  state.selectedApModel = apModel.id;

  applyModelToAccessPoints(apModel.id, true);
  syncAccessPointFrequencies();
  preloadApImages();
  renderDevicePanel();
  renderEditDeviceList();
  invalidateHeatmap();
  saveHistory();
  draw();
  if (document.getElementById('deviceListModal')?.style.display === 'flex') {
    renderDeviceList();
  }

  await customAlert(`AP profile "${apModel.model}" updated and applied to placed APs.`);
});
function openEditWallModal() {
  const modal = document.getElementById('editWallModal');
  modal.style.display = 'flex';
  renderEditWallList();
  selectWallToEdit(state.wallTypes[0].id);
}

function renderEditWallList() {
  const list = document.getElementById('editWallList');
  list.innerHTML = '';
  state.wallTypes.forEach(w => {
    const item = document.createElement('div');
    item.className = 'edit-wall-item';
    item.dataset.id = w.id;
    item.innerHTML = `
      <div class="wall-preview" style="background-color: ${w.color};"></div>
      <span>${w.name}</span>
      <span class="db-val">${w.db5}dB</span>
    `;
    item.onclick = () => selectWallToEdit(w.id);
    list.appendChild(item);
  });
}

let currentEditingId = null;

function selectWallToEdit(id) {
  currentEditingId = id;
  const w = state.wallTypes.find(x => x.id === id);
  if (!w) return;

  document.querySelectorAll('.edit-wall-item').forEach(el => {
    el.classList.toggle('active', el.dataset.id === id);
  });

  document.getElementById('editWallName').value = w.name;
  document.getElementById('editWallThickness').value = w.thickness;
  document.getElementById('editWallMaterial').value = w.id; // Simplified
  document.getElementById('editWallDb24').value = w.db24;
  document.getElementById('editWallDb5').value = w.db5;
  document.getElementById('editWallDb6').value = w.db6;
}

document.getElementById('btnSaveWallType')?.addEventListener('click', async () => {
  const w = state.wallTypes.find(x => x.id === currentEditingId);
  if (w) {
    w.name = document.getElementById('editWallName').value;
    w.thickness = parseInt(document.getElementById('editWallThickness').value);
    w.db24 = parseInt(document.getElementById('editWallDb24').value);
    w.db5 = parseInt(document.getElementById('editWallDb5').value);
    w.db6 = parseInt(document.getElementById('editWallDb6').value);
    renderWallPanel();
    renderEditWallList();
    await customAlert("Wall type saved!");
  }
});

document.querySelector('.add-wall-type-btn')?.addEventListener('click', () => {
  const newId = 'wall_' + Date.now();
  state.wallTypes.push({
    id: newId,
    name: 'New Wall Type',
    db24: 5,
    db5: 10,
    db6: 12,
    thickness: 100,
    color: '#64748b',
    category: 'Materials'
  });
  renderEditWallList();
  selectWallToEdit(newId);
});

document.querySelectorAll('.tool-btn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    setActiveTool(e.currentTarget.dataset.tool, e.currentTarget);
  });
});

document.getElementById('btnPan')?.addEventListener('click', (e) => {
  setActiveTool('pan', e.currentTarget);
});

document.getElementById('uploadBlueprint')?.addEventListener('change', function (e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function (event) {
    state.blueprintImageSrc = event.target.result;
    const img = new Image();
    img.onload = () => {
      state.blueprintImgObj = img;
      draw();
    };
    img.src = state.blueprintImageSrc;
  };
  reader.readAsDataURL(file);
});

// Canvas Interaction
function getMousePos(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left - state.transform.x) / state.transform.scale,
    y: (e.clientY - rect.top - state.transform.y) / state.transform.scale
  };
}

canvas.addEventListener('mousedown', (e) => {
  if (e.button === 1 || state.mode === 'pan') { // Middle click or pan mode
    state.isDragging = true;
    state.lastMousePos = { x: e.clientX, y: e.clientY };
    canvas.style.cursor = 'grabbing';
    return;
  }

  const pos = getMousePos(e);

  // Dragging AP
  if (state.hoveredAp !== null && !state.isDrawing) {
    state.draggingAp = state.hoveredAp;
    return;
  }

  // Dragging wall handle
  if (state.hoveredHandle && !state.isDrawing) {
    state.draggingHandle = state.hoveredHandle;
    return;
  }

  if (state.mode === 'wall' || state.mode === 'scale') {
    if (!state.isDrawing) {
      // Start Drawing
      state.isDrawing = true;
      state.currentLine = [pos, pos];
    } else {
      // Finish Drawing
      if (state.mode === 'scale') {
        finishScaleCalibration();
      } else {
        finishWallDrawing();
      }
    }
  } else if (state.mode === 'ap' && state.apPlacementTab === 'manual') {
    const model = getApModelById(state.selectedApModel);
    const power24 = getDefaultTxPowerForBand(model, 2.4);
    const power5 = getDefaultTxPowerForBand(model, 5);
    // Add AP
    state.accessPoints.push(normalizeAccessPoint({
      name: `AP ${state.accessPoints.length + 1}`,
      model_id: model.id,
      model: model.model,
      vendor: model.vendor,
      x: pos.x / state.pixelsPerMeter,
      y: pos.y / state.pixelsPerMeter,
      z: getDefaultApHeightM(),
      power: getActiveFrequencyGHz() >= 5 ? power5 : power24,
      power24,
      power5,
      gain24: model.gain24,
      gain5: model.gain5,
      frequency: getActiveFrequencyGHz()
    }));
    invalidateHeatmap();
    saveHistory();
    draw();
  } else if (state.mode === 'ap' && state.apPlacementTab === 'auto') {
    const roomIndex = findRoomAtPos(pos);
    if (roomIndex !== null) {
      openEditRoomModal(roomIndex);
    }
    canvas.style.cursor = 'default';
  } else if (state.mode === 'select') {
    // Select Wall
    if (state.hoveredWall !== null) {
      state.selectedElementIndex = state.hoveredWall;
      state.selectedElementType = 'wall';
      state.selectedWallType = state.elements[state.selectedElementIndex].material;

      // Start dragging wall
      state.draggingWall = state.hoveredWall;
      state.lastMousePos = { x: e.clientX, y: e.clientY };
    }
    // Select AP
    else if (state.hoveredAp !== null) {
      state.selectedElementIndex = state.hoveredAp;
      state.selectedElementType = 'ap';
    }
    // Deselect
    else {
      state.selectedElementIndex = null;
      state.selectedElementType = null;
    }
    draw();
  }
});

canvas.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  if (state.isDrawing) {
    state.isDrawing = false;
    state.currentLine = null;
    draw();
  }
});

canvas.addEventListener('mousemove', (e) => {
  if (state.isDragging) {
    const dx = e.clientX - state.lastMousePos.x;
    const dy = e.clientY - state.lastMousePos.y;
    state.transform.x += dx;
    state.transform.y += dy;
    state.lastMousePos = { x: e.clientX, y: e.clientY };
    draw();
    return;
  }

  const pos = getMousePos(e);
  updateLegendHover(pos);

  // Handle AP Dragging
  if (state.draggingAp !== null) {
    const ap = state.accessPoints[state.draggingAp];
    ap.x = pos.x / state.pixelsPerMeter;
    ap.y = pos.y / state.pixelsPerMeter;
    invalidateHeatmap();
    updateLegendHover(pos);
    requestDraw();
    return;
  }

  // Handle Dragging wall handle
  if (state.draggingHandle) {
    const elem = state.elements[state.draggingHandle.elementIndex];
    elem.points[state.draggingHandle.pointIndex] = [pos.x / state.pixelsPerMeter, pos.y / state.pixelsPerMeter];
    invalidateHeatmap();
    updateLegendHover(pos);
    requestDraw();
    return;
  }

  // Handle Dragging entire wall
  if (state.draggingWall !== null) {
    const dx = (e.clientX - state.lastMousePos.x) / (state.transform.scale * state.pixelsPerMeter);
    const dy = (e.clientY - state.lastMousePos.y) / (state.transform.scale * state.pixelsPerMeter);

    const wall = state.elements[state.draggingWall];
    wall.points[0][0] += dx;
    wall.points[0][1] += dy;
    wall.points[1][0] += dx;
    wall.points[1][1] += dy;

    state.lastMousePos = { x: e.clientX, y: e.clientY };
    invalidateHeatmap();
    updateLegendHover(pos);
    requestDraw();
    return;
  }

  // Handle Hover Detection
  if (!state.isDrawing) {
    if (state.mode === 'wall' || state.mode === 'select') {
      state.hoveredHandle = findHandleAtPos(pos);
      state.hoveredWall = state.hoveredHandle ? null : findWallAtPos(pos);
    } else {
      state.hoveredHandle = null;
      state.hoveredWall = null;
    }

    if (state.mode === 'ap' || state.mode === 'select') {
      state.hoveredAp = findApAtPos(pos);
    } else {
      state.hoveredAp = null;
    }

    // Determine cursor
    if (state.draggingHandle || state.draggingAp !== null) {
      canvas.style.cursor = 'move';
    } else if (state.hoveredHandle || state.hoveredAp !== null) {
      canvas.style.cursor = 'move';
    } else if (state.hoveredWall !== null) {
      canvas.style.cursor = 'pointer';
    } else if (state.mode === 'select') {
      canvas.style.cursor = 'default';
    } else if (state.mode === 'ap' && state.apPlacementTab === 'auto') {
      canvas.style.cursor = findRoomAtPos(pos) !== null ? 'pointer' : 'default';
    } else if (state.mode === 'pan') {
      canvas.style.cursor = 'grab';
    } else {
      canvas.style.cursor = 'crosshair';
    }
    draw();
  }

  if (state.isDrawing && (state.mode === 'wall' || state.mode === 'scale')) {
    state.currentLine[1] = pos;
    draw();
  }
});

canvas.addEventListener('mouseleave', () => {
  resetLegendHover();
});

canvas.addEventListener('mouseup', async () => {
  let shouldFinalizeHeatmap = false;

  if (state.isDragging) {
    state.isDragging = false;
    canvas.style.cursor = 'crosshair';
  }
  if (state.draggingHandle) {
    state.draggingHandle = null;
    invalidateHeatmap();
    shouldFinalizeHeatmap = true;
    saveHistory();
  }
  if (state.draggingAp !== null) {
    state.draggingAp = null;
    invalidateHeatmap();
    shouldFinalizeHeatmap = true;
    saveHistory();
  }
  if (state.draggingWall !== null) {
    state.draggingWall = null;
    invalidateHeatmap();
    shouldFinalizeHeatmap = true;
    saveHistory();
  }

  if (shouldFinalizeHeatmap) {
    draw();
  }
});

async function finishScaleCalibration() {
  state.isDrawing = false;
  const dx = state.currentLine[1].x - state.currentLine[0].x;
  const dy = state.currentLine[1].y - state.currentLine[0].y;
  const pixelDist = Math.sqrt(dx * dx + dy * dy);
  if (pixelDist > 5) {
    const actualDist = await customPrompt("Enter the actual length of the line you drew (in meters):", "1.0");
    if (actualDist && !isNaN(actualDist)) {
      state.pixelsPerMeter = pixelDist / parseFloat(actualDist);
      state.isScaleSet = true;
      await customAlert(`Scale set: 1 meter = ${state.pixelsPerMeter.toFixed(2)} pixels`);
      const wallBtn = document.querySelector(".tool-btn[data-tool=\"wall\"]");
      if (wallBtn) setActiveTool("wall", wallBtn);
    }
  }
  state.currentLine = null;
  draw();
}

function finishWallDrawing() {
  const type = state.mode;
  state.isDrawing = false;
  if (state.currentLine) {
    const selectedWall = state.wallTypes.find(w => w.id === state.selectedWallType);
    state.elements.push({
      type: type,
      points: [
        [state.currentLine[0].x / state.pixelsPerMeter, state.currentLine[0].y / state.pixelsPerMeter],
        [state.currentLine[1].x / state.pixelsPerMeter, state.currentLine[1].y / state.pixelsPerMeter]
      ],
      material: type === 'wall' ? selectedWall.id : null,
      properties: type === 'wall' ? selectedWall : null
    });
    state.currentLine = null;
    invalidateHeatmap();
    saveHistory();
    draw();
  }
}

// Zooming
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const zoomIntensity = 0.1;
  const wheel = e.deltaY < 0 ? 1 : -1;
  const zoom = Math.exp(wheel * zoomIntensity);

  const rect = canvas.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  state.transform.x = mouseX - (mouseX - state.transform.x) * zoom;
  state.transform.y = mouseY - (mouseY - state.transform.y) * zoom;
  state.transform.scale *= zoom;

  draw();
});

// View Controls
document.getElementById('btnUndo')?.addEventListener('click', undo);
document.getElementById('btnRedo')?.addEventListener('click', redo);

document.getElementById('btnZoomIn').addEventListener('click', () => zoom(1.2));
document.getElementById('btnZoomOut').addEventListener('click', () => zoom(0.8));
document.getElementById('btnFit').addEventListener('click', () => {
  state.transform = { x: 0, y: 0, scale: 1 };
  draw();
});

function zoom(factor) {
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  state.transform.x = cx - (cx - state.transform.x) * factor;
  state.transform.y = cy - (cy - state.transform.y) * factor;
  state.transform.scale *= factor;
  draw();
}
// ==================== WIFI HEATMAP ENGINE ====================
// Physics-based WiFi signal propagation with wall attenuation

let heatmapEnabled = true;
let heatmapCache = null;   // Cached ImageData
let heatmapDirty = true;   // Flag to recompute
let heatmapWorldBounds = null;

// Invalidate heatmap when walls or APs change
function invalidateHeatmap() {
  heatmapDirty = true;
  heatmapCache = null;
  heatmapWorldBounds = null;
}

function setHeatmapEnabled(enabled, shouldInvalidate = true) {
  state.heatmapEnabled = Boolean(enabled);
  document.getElementById('btnToggleHeatmap')?.classList.toggle('active', state.heatmapEnabled);
  if (state.heatmapEnabled && shouldInvalidate) {
    invalidateHeatmap();
  }
}

function isHeatmapInteractiveMove() {
  return state.draggingAp !== null || state.draggingHandle || state.draggingWall !== null;
}

/**
 * Indoor Path Loss Model (ITU-R P.1238)
 * PL(d) = 20*log10(f_MHz) + N*log10(d_m) - 28
 * 
 * N (distance power loss coefficient):
 *   - 28 for residential (few walls)
 *   - 30 for office environment  
 *   - 33 for commercial/dense office
 * 
 * For 2.4 GHz (2400 MHz), PL_ref = 20*log10(2400) - 28 = 67.6 - 28 = 39.6 dB at 1m
 */
function indoorPathLoss(distanceM, freqGHz) {
  if (distanceM < 0.1) distanceM = 0.1;
  const freqMHz = freqGHz * 1000;
  const profile = getSimulationProfile(freqGHz);
  return 20 * Math.log10(freqMHz) + 10 * profile.pathLossExponent * Math.log10(distanceM) - 28;
}

/**
 * Check if a line segment (ray from AP to point) intersects a wall segment.
 */
function segmentIntersectionParameter(ax, ay, bx, by, cx, cy, dx, dy) {
  const denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
  if (Math.abs(denom) < 1e-10) return null;
  const t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom;
  const u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom;
  const epsilon = 1e-6;
  return t > epsilon && t < 1 - epsilon && u >= -epsilon && u <= 1 + epsilon ? t : null;
}

function segmentsIntersect(ax, ay, bx, by, cx, cy, dx, dy) {
  return segmentIntersectionParameter(ax, ay, bx, by, cx, cy, dx, dy) !== null;
}

/**
 * Sum wall attenuation along a ray. Coordinates in METERS.
 */
function countWallAttenuation(x1, y1, x2, y2, walls, freqBand) {
  return traceWallAttenuation(x1, y1, x2, y2, walls, freqBand).totalDb;
}

function traceWallAttenuation(x1, y1, x2, y2, walls, freqBand) {
  const profile = BAND_SIMULATION_PROFILES[freqBand] || BAND_SIMULATION_PROFILES['2.4ghz'];
  const crossings = new Map();
  for (const wall of walls) {
    const t = segmentIntersectionParameter(x1, y1, x2, y2, wall.p1x, wall.p1y, wall.p2x, wall.p2y);
    if (t === null) continue;
    const key = Math.round(t * 100000);
    const loss = ((freqBand === '5ghz') ? wall.db5 : wall.db24) * profile.wallLossMultiplier;
    crossings.set(key, Math.max(crossings.get(key) || 0, loss));
  }
  return {
    count: crossings.size,
    totalDb: [...crossings.values()].reduce((total, loss) => total + loss, 0)
  };
}

function getWallSegments() {
  const walls = [];
  state.elements.forEach(elem => {
    if (elem.type === 'wall' && elem.points.length === 2) {
      const wt = state.wallTypes.find(w => w.id === elem.material) || state.wallTypes[0];
      walls.push({
        p1x: elem.points[0][0],
        p1y: elem.points[0][1],
        p2x: elem.points[1][0],
        p2y: elem.points[1][1],
        db24: wt.db24,
        db5: wt.db5
      });
    }
  });
  return walls;
}

/**
 * Calculate signal strength (dBm) at a point from a single AP.
 * Uses ITU Indoor Model + Multi-Wall + Fade Margin.
 */
function calcSignalAt(px, py, ap, walls) {
  const apModel = getApModelById(ap.model_id || state.selectedApModel);
  const freqGHz = getActiveFrequencyGHz();
  const txPower = getTxPowerForBand(apModel, ap, freqGHz);
  const antennaGain = getAntennaGainForBand(apModel, ap, freqGHz);
  const profile = getSimulationProfile(freqGHz);

  const dx = px - ap.x;
  const dy = py - ap.y;
  const dz = Number(ap.z ?? getDefaultApHeightM()) - getClientHeightM();
  const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

  if (dist < 0.1) return txPower;

  const pathLoss = indoorPathLoss(dist, freqGHz);
  const wallLoss = countWallAttenuation(ap.x, ap.y, px, py, walls, getActiveBandKey());

  // Signal = TX - PathLoss - WallLoss - FadeMargin
  const signal = txPower + antennaGain - pathLoss - wallLoss - profile.fadeMarginDb;

  return signal;

}

function estimateSignalRadiusM(ap, targetDbm) {
  const apModel = getApModelById(ap.model_id || state.selectedApModel);
  const freqGHz = getActiveFrequencyGHz();
  const txPower = getTxPowerForBand(apModel, ap, freqGHz);
  const antennaGain = getAntennaGainForBand(apModel, ap, freqGHz);
  const profile = getSimulationProfile(freqGHz);
  const freqMHz = freqGHz * 1000;
  const allowedPathLoss = txPower + antennaGain - profile.fadeMarginDb - targetDbm;
  const exponent = (allowedPathLoss - (20 * Math.log10(freqMHz)) + 28) / (10 * profile.pathLossExponent);
  return clamp(Math.pow(10, exponent), 1.5, 80);
}

/**
 * Heatmap palette tuned to match the planner mockup:
 * strong green indoors, yellow transition, orange/pink at weak edges.
 */
function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function lerp(start, end, t) {
  return start + (end - start) * t;
}

function smoothstep(edge0, edge1, value) {
  const t = clamp((value - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

function signalToColor(dbm) {
  const range = getLegendSignalRange();
  const stops = [
    { dbm: range.poor - 8, color: [255, 153, 170], alpha: 0 },
    { dbm: range.poor, color: [255, 153, 170], alpha: 155 },
    { dbm: range.mid, color: [255, 186, 107], alpha: 170 },
    { dbm: range.good - 0.1, color: [255, 232, 84], alpha: 188 },
    { dbm: range.good, color: [155, 219, 92], alpha: 198 },
    { dbm: range.good + 18, color: [148, 218, 92], alpha: 210 }
  ];

  if (dbm <= stops[0].dbm) {
    return [0, 0, 0, 0];
  }

  for (let i = 1; i < stops.length; i++) {
    const low = stops[i - 1];
    const high = stops[i];
    if (dbm <= high.dbm) {
      const t = smoothstep(low.dbm, high.dbm, dbm);
      return [
        Math.round(lerp(low.color[0], high.color[0], t)),
        Math.round(lerp(low.color[1], high.color[1], t)),
        Math.round(lerp(low.color[2], high.color[2], t)),
        Math.round(lerp(low.alpha, high.alpha, t))
      ];
    }
  }

  const strongest = stops[stops.length - 1];
  return [...strongest.color, strongest.alpha];
}

function distancePointToSegment(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;

  if (dx === 0 && dy === 0) {
    return Math.hypot(px - x1, py - y1);
  }

  const t = clamp(((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy), 0, 1);
  const projX = x1 + t * dx;
  const projY = y1 + t * dy;
  return Math.hypot(px - projX, py - projY);
}

function wallElementToSegment(elem) {
  if (!elem || elem.type !== 'wall' || !Array.isArray(elem.points) || elem.points.length < 2) {
    return null;
  }

  return {
    p1x: Number(elem.points[0][0]),
    p1y: Number(elem.points[0][1]),
    p2x: Number(elem.points[1][0]),
    p2y: Number(elem.points[1][1])
  };
}

function wallSegmentLength(seg) {
  return Math.hypot(seg.p2x - seg.p1x, seg.p2y - seg.p1y);
}

function wallSegmentAngle(seg) {
  return Math.atan2(seg.p2y - seg.p1y, seg.p2x - seg.p1x);
}

function wallAngleDiff(a, b) {
  const diff = Math.abs(a - b) % Math.PI;
  return Math.min(diff, Math.PI - diff);
}

function wallOverlapRatio(a, b) {
  const aHorizontal = Math.abs(a.p2x - a.p1x) >= Math.abs(a.p2y - a.p1y);
  const aMin = aHorizontal ? Math.min(a.p1x, a.p2x) : Math.min(a.p1y, a.p2y);
  const aMax = aHorizontal ? Math.max(a.p1x, a.p2x) : Math.max(a.p1y, a.p2y);
  const bMin = aHorizontal ? Math.min(b.p1x, b.p2x) : Math.min(b.p1y, b.p2y);
  const bMax = aHorizontal ? Math.max(b.p1x, b.p2x) : Math.max(b.p1y, b.p2y);
  const overlap = Math.max(0, Math.min(aMax, bMax) - Math.max(aMin, bMin));
  const shorter = Math.max(0.01, Math.min(aMax - aMin, bMax - bMin));
  return overlap / shorter;
}

function isDuplicateWallSegment(candidate, existing, options = {}) {
  const angleTolerance = options.angleTolerance ?? (8 * Math.PI / 180);
  const lineToleranceM = options.lineToleranceM ?? 0.28;
  const endpointToleranceM = options.endpointToleranceM ?? 0.32;
  const overlapThreshold = options.overlapThreshold ?? 0.68;

  const candidateLength = wallSegmentLength(candidate);
  const existingLength = wallSegmentLength(existing);
  if (candidateLength < 0.05 || existingLength < 0.05) return false;

  const sameEndpointOrder =
    Math.hypot(candidate.p1x - existing.p1x, candidate.p1y - existing.p1y) <= endpointToleranceM &&
    Math.hypot(candidate.p2x - existing.p2x, candidate.p2y - existing.p2y) <= endpointToleranceM;
  const reversedEndpointOrder =
    Math.hypot(candidate.p1x - existing.p2x, candidate.p1y - existing.p2y) <= endpointToleranceM &&
    Math.hypot(candidate.p2x - existing.p1x, candidate.p2y - existing.p1y) <= endpointToleranceM;
  if (sameEndpointOrder || reversedEndpointOrder) return true;

  if (wallAngleDiff(wallSegmentAngle(candidate), wallSegmentAngle(existing)) > angleTolerance) {
    return false;
  }

  const candidateLineDistance = Math.max(
    distancePointToSegment(candidate.p1x, candidate.p1y, existing.p1x, existing.p1y, existing.p2x, existing.p2y),
    distancePointToSegment(candidate.p2x, candidate.p2y, existing.p1x, existing.p1y, existing.p2x, existing.p2y)
  );
  const existingLineDistance = Math.max(
    distancePointToSegment(existing.p1x, existing.p1y, candidate.p1x, candidate.p1y, candidate.p2x, candidate.p2y),
    distancePointToSegment(existing.p2x, existing.p2y, candidate.p1x, candidate.p1y, candidate.p2x, candidate.p2y)
  );
  const lineDistance = Math.min(candidateLineDistance, existingLineDistance);

  return lineDistance <= lineToleranceM && wallOverlapRatio(candidate, existing) >= overlapThreshold;
}

function hasDuplicateWall(candidateElement, existingElements) {
  const candidate = wallElementToSegment(candidateElement);
  if (!candidate) return false;

  return existingElements.some(elem => {
    const existing = wallElementToSegment(elem);
    return existing && isDuplicateWallSegment(candidate, existing);
  });
}

function getHeatmapBounds() {
  const ppm = state.pixelsPerMeter || 30.7;

  if (state.blueprintImgObj) {
    return {
      minX: 0,
      minY: 0,
      widthM: state.blueprintImgObj.width / ppm,
      heightM: state.blueprintImgObj.height / ppm
    };
  }

  const xs = [];
  const ys = [];

  state.elements.forEach(elem => {
    elem.points.forEach(point => {
      xs.push(point[0]);
      ys.push(point[1]);
    });
  });

  state.accessPoints.forEach(ap => {
    xs.push(ap.x);
    ys.push(ap.y);
  });

  if (!xs.length || !ys.length) {
    return null;
  }

  const paddingM = 1.5;
  const minX = Math.max(0, Math.min(...xs) - paddingM);
  const minY = Math.max(0, Math.min(...ys) - paddingM);
  const maxX = Math.max(...xs) + paddingM;
  const maxY = Math.max(...ys) + paddingM;

  return {
    minX,
    minY,
    widthM: Math.max(4, maxX - minX),
    heightM: Math.max(4, maxY - minY)
  };
}

function polygonArea(points) {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const current = points[i];
    const next = points[(i + 1) % points.length];
    area += current.x * next.y - next.x * current.y;
  }
  return Math.abs(area) / 2;
}

function polygonCentroid(points) {
  let areaFactor = 0;
  let cx = 0;
  let cy = 0;

  for (let i = 0; i < points.length; i++) {
    const current = points[i];
    const next = points[(i + 1) % points.length];
    const cross = current.x * next.y - next.x * current.y;
    areaFactor += cross;
    cx += (current.x + next.x) * cross;
    cy += (current.y + next.y) * cross;
  }

  if (Math.abs(areaFactor) < 1e-8) {
    return {
      x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
      y: points.reduce((sum, point) => sum + point.y, 0) / points.length
    };
  }

  return {
    x: cx / (3 * areaFactor),
    y: cy / (3 * areaFactor)
  };
}

function convexHull(points) {
  const unique = [...new Map(points.map(point => [`${point.x.toFixed(4)},${point.y.toFixed(4)}`, point])).values()]
    .sort((a, b) => a.x - b.x || a.y - b.y);
  if (unique.length <= 2) return unique;

  const cross = (origin, a, b) =>
    (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
  const lower = [];
  unique.forEach(point => {
    while (lower.length >= 2 && cross(lower.at(-2), lower.at(-1), point) <= 0) lower.pop();
    lower.push(point);
  });
  const upper = [];
  [...unique].reverse().forEach(point => {
    while (upper.length >= 2 && cross(upper.at(-2), upper.at(-1), point) <= 0) upper.pop();
    upper.push(point);
  });
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function generateServiceBoundaryFromWalls() {
  const footprint = detectBlueprintFootprintBoundary();
  if (footprint?.length >= 3) return footprint;
  const points = getWallSegments().flatMap(wall => [
    { x: wall.p1x, y: wall.p1y },
    { x: wall.p2x, y: wall.p2y }
  ]);
  return convexHull(points);
}

function buildRowEnvelopePolygon(component, cellSize) {
  const rows = new Map();
  component.forEach(cell => {
    const range = rows.get(cell.y) || { min: cell.x, max: cell.x };
    range.min = Math.min(range.min, cell.x);
    range.max = Math.max(range.max, cell.x);
    rows.set(cell.y, range);
  });
  const ordered = [...rows.entries()]
    .map(([y, range]) => ({ y, ...range }))
    .sort((a, b) => a.y - b.y);
  if (ordered.length < 3) return null;

  const quantize = value => Math.round(value / 2) * 2;
  ordered.forEach(row => {
    row.min = quantize(row.min);
    row.max = quantize(row.max + 1);
  });

  const left = [{ x: ordered[0].min, y: ordered[0].y }];
  for (let index = 1; index < ordered.length; index++) {
    const previous = ordered[index - 1];
    const current = ordered[index];
    if (current.y > previous.y + 1 || current.min !== previous.min) {
      left.push({ x: previous.min, y: current.y });
      left.push({ x: current.min, y: current.y });
    }
  }
  const last = ordered.at(-1);
  left.push({ x: last.min, y: last.y + 1 });

  const right = [{ x: last.max, y: last.y + 1 }];
  for (let index = ordered.length - 2; index >= 0; index--) {
    const current = ordered[index];
    const below = ordered[index + 1];
    if (below.y > current.y + 1 || current.max !== below.max) {
      right.push({ x: below.max, y: below.y });
      right.push({ x: current.max, y: below.y });
    }
  }
  right.push({ x: ordered[0].max, y: ordered[0].y });

  const polygon = simplifyGridPolygon([...left, ...right]).map(point => ({
    x: point.x * cellSize,
    y: point.y * cellSize
  }));
  return isValidOrthogonalRoomPolygon(polygon) ? polygon : null;
}

function detectBlueprintFootprintBoundary() {
  const analysis = getBlueprintAnalysisData();
  const ppm = Number(state.pixelsPerMeter || 0);
  if (!analysis || ppm <= 0 || !blueprintHasFilledWallStyle()) return null;

  const cellSize = clamp(Math.max(analysis.width, analysis.height) / ppm / 280, 0.12, 0.28);
  const gridW = Math.max(8, Math.ceil(analysis.width / ppm / cellSize));
  const gridH = Math.max(8, Math.ceil(analysis.height / ppm / cellSize));
  const occupied = new Uint8Array(gridW * gridH);

  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const pixelX = (gx + 0.5) * cellSize * ppm;
      const pixelY = (gy + 0.5) * cellSize * ppm;
      let nonWhite = 0;
      let samples = 0;
      const radius = Math.max(1, Math.round(cellSize * ppm * 0.35));
      for (const offsetY of [-radius, 0, radius]) {
        for (const offsetX of [-radius, 0, radius]) {
          samples++;
          if (blueprintPixelBrightness(analysis, pixelX + offsetX, pixelY + offsetY) < 251) nonWhite++;
        }
      }
      if (nonWhite >= Math.ceil(samples * 0.55)) occupied[gy * gridW + gx] = 1;
    }
  }

  // Filled floor plans may contain white furniture cut-outs and text. Collapse
  // each meaningful row to its outer occupied span so the footprint follows
  // the building envelope instead of every interior drawing detail.
  for (let gy = 0; gy < gridH; gy++) {
    const occupiedXs = [];
    for (let gx = 0; gx < gridW; gx++) {
      if (occupied[gy * gridW + gx]) occupiedXs.push(gx);
    }
    if (occupiedXs.length < Math.max(4, gridW * 0.08)) continue;
    const minX = Math.min(...occupiedXs);
    const maxX = Math.max(...occupiedXs);
    if (maxX - minX < gridW * 0.12) continue;
    for (let gx = minX; gx <= maxX; gx++) occupied[gy * gridW + gx] = 1;
  }

  const visited = new Uint8Array(gridW * gridH);
  let largest = [];
  for (let sy = 0; sy < gridH; sy++) {
    for (let sx = 0; sx < gridW; sx++) {
      const start = sy * gridW + sx;
      if (!occupied[start] || visited[start]) continue;
      const stack = [{ x: sx, y: sy }];
      const component = [];
      visited[start] = 1;
      while (stack.length) {
        const cell = stack.pop();
        component.push(cell);
        for (const direction of [{ x: 1, y: 0 }, { x: -1, y: 0 }, { x: 0, y: 1 }, { x: 0, y: -1 }]) {
          const nx = cell.x + direction.x;
          const ny = cell.y + direction.y;
          if (nx < 0 || ny < 0 || nx >= gridW || ny >= gridH) continue;
          const index = ny * gridW + nx;
          if (!occupied[index] || visited[index]) continue;
          visited[index] = 1;
          stack.push({ x: nx, y: ny });
        }
      }
      if (component.length > largest.length) largest = component;
    }
  }

  const occupiedRatio = largest.length / Math.max(1, gridW * gridH);
  if (occupiedRatio < 0.12 || occupiedRatio > 0.92) return null;
  const polygon = buildRowEnvelopePolygon(largest, cellSize);
  if (!polygon || polygon.length < 4 || polygon.length > 80) return null;
  return polygon;
}

function ensureExteriorBoundaryWalls() {
  const boundary = detectBlueprintFootprintBoundary();
  if (!boundary?.length) return 0;
  const material = state.selectedWallType || 'common_brick';
  let added = 0;

  boundary.forEach((point, index) => {
    const next = boundary[(index + 1) % boundary.length];
    const length = Math.hypot(next.x - point.x, next.y - point.y);
    if (length < 0.45) return;
    const span = normalizeWallSpan({
      p1x: point.x, p1y: point.y, p2x: next.x, p2y: next.y
    });
    if (!span) return;
    const existing = getWallSegments().map(normalizeWallSpan).filter(Boolean);
    const aligned = existing.filter(item =>
      item.orientation === span.orientation && Math.abs(item.fixed - span.fixed) <= 0.28
    );
    const coverage = getMergedCoverageLength(aligned, span.min, span.max, 0.3)
      / Math.max(0.01, span.max - span.min);
    if (coverage >= 0.82) return;

    const covered = aligned
      .map(item => ({ min: Math.max(span.min, item.min), max: Math.min(span.max, item.max) }))
      .filter(item => item.max > item.min)
      .sort((a, b) => a.min - b.min);
    const merged = [];
    covered.forEach(item => {
      const last = merged.at(-1);
      if (last && item.min <= last.max + 0.3) last.max = Math.max(last.max, item.max);
      else merged.push({ ...item });
    });

    const gaps = [];
    let cursor = span.min;
    merged.forEach(item => {
      if (item.min - cursor >= 0.4) gaps.push({ min: cursor, max: item.min });
      cursor = Math.max(cursor, item.max);
    });
    if (span.max - cursor >= 0.4) gaps.push({ min: cursor, max: span.max });

    gaps.forEach(gap => {
      const points = span.orientation === 'h'
        ? [[gap.min, span.fixed], [gap.max, span.fixed]]
        : [[span.fixed, gap.min], [span.fixed, gap.max]];
      const candidate = { type: 'wall', points, material };
      if (hasDuplicateWall(candidate, state.elements)) return;
      state.elements.push(candidate);
      added++;
    });
  });

  if (added > 0) invalidateHeatmap();
  return added;
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects = ((yi > point.y) !== (yj > point.y)) &&
      (point.x < ((xj - xi) * (point.y - yi)) / ((yj - yi) || 1e-9) + xi);
    if (intersects) inside = !inside;
  }
  return inside;
}

function findRoomAtPos(pos) {
  if (!state.showRoomsOverlay || state.apPlacementTab !== 'auto') return null;
  const point = {
    x: pos.x / state.pixelsPerMeter,
    y: pos.y / state.pixelsPerMeter
  };

  for (let i = (state.rooms || []).length - 1; i >= 0; i--) {
    const polygon = state.rooms[i]?.polygon;
    if (Array.isArray(polygon) && polygon.length >= 3 && pointInPolygon(point, polygon)) {
      return i;
    }
  }

  return null;
}

function simplifyGridPolygon(points) {
  if (points.length <= 3) return points;
  const simplified = [];

  for (let i = 0; i < points.length; i++) {
    const prev = points[(i - 1 + points.length) % points.length];
    const current = points[i];
    const next = points[(i + 1) % points.length];
    const sameX = prev.x === current.x && current.x === next.x;
    const sameY = prev.y === current.y && current.y === next.y;
    if (!sameX && !sameY) {
      simplified.push(current);
    }
  }

  return simplified.length >= 3 ? simplified : points;
}

function buildComponentPolygon(component, componentSet, bounds, cellSize, gridW, gridH) {
  const edges = [];
  const addEdge = (x1, y1, x2, y2) => {
    edges.push({
      from: `${x1},${y1}`,
      to: `${x2},${y2}`,
      x1,
      y1,
      x2,
      y2,
      used: false
    });
  };

  component.forEach(cell => {
    const x = cell.x;
    const y = cell.y;
    if (y === 0 || !componentSet.has(`${x},${y - 1}`)) addEdge(x, y, x + 1, y);
    if (x === gridW - 1 || !componentSet.has(`${x + 1},${y}`)) addEdge(x + 1, y, x + 1, y + 1);
    if (y === gridH - 1 || !componentSet.has(`${x},${y + 1}`)) addEdge(x + 1, y + 1, x, y + 1);
    if (x === 0 || !componentSet.has(`${x - 1},${y}`)) addEdge(x, y + 1, x, y);
  });

  const byStart = new Map();
  edges.forEach((edge, index) => {
    if (!byStart.has(edge.from)) byStart.set(edge.from, []);
    byStart.get(edge.from).push(index);
  });

  let bestLoop = [];
  edges.forEach(edge => {
    if (edge.used) return;
    const loop = [{ x: edge.x1, y: edge.y1 }];
    let current = edge;
    let closed = false;
    current.used = true;

    for (let guard = 0; guard < edges.length + 5; guard++) {
      loop.push({ x: current.x2, y: current.y2 });
      if (current.to === edge.from) {
        closed = true;
        break;
      }
      const nextIndex = (byStart.get(current.to) || []).find(index => !edges[index].used);
      if (nextIndex === undefined) break;
      current = edges[nextIndex];
      current.used = true;
    }

    if (closed && loop.length > bestLoop.length) {
      bestLoop = loop;
    }
  });

  if (bestLoop.length < 4) return null;
  const polygon = simplifyGridPolygon(bestLoop).map(point => ({
    x: bounds.minX + point.x * cellSize,
    y: bounds.minY + point.y * cellSize
  }));

  return isValidOrthogonalRoomPolygon(polygon) ? polygon : null;
}

function isValidOrthogonalRoomPolygon(polygon) {
  if (!Array.isArray(polygon) || polygon.length < 4) return false;
  let perimeter = 0;
  let diagonalLength = 0;

  for (let index = 0; index < polygon.length; index++) {
    const current = polygon[index];
    const next = polygon[(index + 1) % polygon.length];
    const dx = Math.abs(next.x - current.x);
    const dy = Math.abs(next.y - current.y);
    const length = Math.hypot(dx, dy);
    perimeter += length;
    if (dx > 0.05 && dy > 0.05) diagonalLength += length;
  }

  return perimeter > 0 && diagonalLength / perimeter < 0.02;
}

function isLikelyEnclosedRoomPolygon(polygon) {
  if (!isValidOrthogonalRoomPolygon(polygon) || polygon.length > 20) return false;
  const xs = polygon.map(point => point.x);
  const ys = polygon.map(point => point.y);
  const width = Math.max(...xs) - Math.min(...xs);
  const height = Math.max(...ys) - Math.min(...ys);
  if (width < 0.8 || height < 0.8) return false;
  const aspect = Math.max(width / height, height / width);
  const boundingArea = width * height;
  const fillRatio = polygonArea(polygon) / Math.max(0.001, boundingArea);
  const complexityPenalty = Math.max(0, polygon.length - 8) * 0.012;
  const requiredFillRatio = Math.max(0.42, 0.58 - complexityPenalty);
  return aspect <= 5.5 && fillRatio >= requiredFillRatio;
}

function markBlockedSegment(blocked, gridW, gridH, bounds, cellSize, x1, y1, x2, y2, radiusM) {
  const minX = Math.max(0, Math.floor((Math.min(x1, x2) - radiusM - bounds.minX) / cellSize));
  const maxX = Math.min(gridW - 1, Math.ceil((Math.max(x1, x2) + radiusM - bounds.minX) / cellSize));
  const minY = Math.max(0, Math.floor((Math.min(y1, y2) - radiusM - bounds.minY) / cellSize));
  const maxY = Math.min(gridH - 1, Math.ceil((Math.max(y1, y2) + radiusM - bounds.minY) / cellSize));

  for (let gy = minY; gy <= maxY; gy++) {
    for (let gx = minX; gx <= maxX; gx++) {
      const px = bounds.minX + (gx + 0.5) * cellSize;
      const py = bounds.minY + (gy + 0.5) * cellSize;
      if (distancePointToSegment(px, py, x1, y1, x2, y2) <= radiusM) {
        blocked[gy * gridW + gx] = 1;
      }
    }
  }
}

function getWallEndpointData(walls) {
  const endpoints = [];
  walls.forEach(wall => {
    const dx = wall.p2x - wall.p1x;
    const dy = wall.p2y - wall.p1y;
    const length = Math.hypot(dx, dy);
    if (length < 0.1) return;
    const dir = { x: dx / length, y: dy / length };
    endpoints.push({ x: wall.p1x, y: wall.p1y, dir });
    endpoints.push({ x: wall.p2x, y: wall.p2y, dir });
  });
  return endpoints;
}

function getWallDetectionBounds(walls) {
  if (!walls.length) return getHeatmapBounds();

  const xs = [];
  const ys = [];
  walls.forEach(wall => {
    xs.push(wall.p1x, wall.p2x);
    ys.push(wall.p1y, wall.p2y);
  });

  const paddingM = 1.0;
  const minX = Math.max(0, Math.min(...xs) - paddingM);
  const minY = Math.max(0, Math.min(...ys) - paddingM);
  const maxX = Math.max(...xs) + paddingM;
  const maxY = Math.max(...ys) + paddingM;

  return {
    minX,
    minY,
    widthM: Math.max(4, maxX - minX),
    heightM: Math.max(4, maxY - minY)
  };
}

function isMostlyHorizontal(wall) {
  return Math.abs(wall.p2x - wall.p1x) >= Math.abs(wall.p2y - wall.p1y) * 2.2;
}

function isMostlyVertical(wall) {
  return Math.abs(wall.p2y - wall.p1y) >= Math.abs(wall.p2x - wall.p1x) * 2.2;
}

function normalizeWallSpan(wall) {
  if (isMostlyHorizontal(wall)) {
    return {
      orientation: 'h',
      fixed: (wall.p1y + wall.p2y) / 2,
      min: Math.min(wall.p1x, wall.p2x),
      max: Math.max(wall.p1x, wall.p2x)
    };
  }

  if (isMostlyVertical(wall)) {
    return {
      orientation: 'v',
      fixed: (wall.p1x + wall.p2x) / 2,
      min: Math.min(wall.p1y, wall.p2y),
      max: Math.max(wall.p1y, wall.p2y)
    };
  }

  return null;
}

function detectDoorOpenings(walls) {
  const spans = walls.map(normalizeWallSpan).filter(Boolean);
  const groups = [];
  spans.forEach(span => {
    let group = groups.find(item =>
      item.orientation === span.orientation && Math.abs(item.fixed - span.fixed) <= 0.24
    );
    if (!group) {
      group = { orientation: span.orientation, fixed: span.fixed, spans: [] };
      groups.push(group);
    }
    group.spans.push(span);
    group.fixed = (group.fixed * (group.spans.length - 1) + span.fixed) / group.spans.length;
  });

  const openings = [];
  groups.forEach(group => {
    const ordered = [...group.spans].sort((a, b) => a.min - b.min);
    for (let index = 0; index < ordered.length - 1; index++) {
      const before = ordered[index];
      const after = ordered[index + 1];
      const gapStart = before.max;
      const gapEnd = after.min;
      const gap = gapEnd - gapStart;
      const beforeLength = before.max - before.min;
      const afterLength = after.max - after.min;
      if (gap < 0.55 || gap > 1.8 || beforeLength < 0.35 || afterLength < 0.35) continue;

      const opening = group.orientation === 'h'
        ? { x1: gapStart, y1: group.fixed, x2: gapEnd, y2: group.fixed, orientation: 'h', widthM: gap }
        : { x1: group.fixed, y1: gapStart, x2: group.fixed, y2: gapEnd, orientation: 'v', widthM: gap };
      const duplicate = openings.some(existing =>
        existing.orientation === opening.orientation
        && Math.hypot(existing.x1 - opening.x1, existing.y1 - opening.y1) < 0.3
        && Math.hypot(existing.x2 - opening.x2, existing.y2 - opening.y2) < 0.3
      );
      if (!duplicate) openings.push(opening);
    }
  });
  return openings;
}

function spanCovers(span, start, end, toleranceM = 0.35) {
  return span.min <= start + toleranceM && span.max >= end - toleranceM;
}

function getMergedCoverageLength(spans, start, end, gapToleranceM = 0.35) {
  const clipped = spans
    .map(span => ({
      min: Math.max(start, span.min),
      max: Math.min(end, span.max)
    }))
    .filter(span => span.max > span.min)
    .sort((a, b) => a.min - b.min);

  if (!clipped.length) return 0;

  let covered = 0;
  let currentMin = clipped[0].min;
  let currentMax = clipped[0].max;

  for (let i = 1; i < clipped.length; i++) {
    const span = clipped[i];
    if (span.min <= currentMax + gapToleranceM) {
      currentMax = Math.max(currentMax, span.max);
    } else {
      covered += currentMax - currentMin;
      currentMin = span.min;
      currentMax = span.max;
    }
  }

  covered += currentMax - currentMin;
  return covered;
}

function axisCoverageRatio(spans, fixedValue, start, end, fixedToleranceM, gapToleranceM = 0.35) {
  const relevant = spans.filter(span => Math.abs(span.fixed - fixedValue) <= fixedToleranceM);
  const length = Math.max(0.001, end - start);
  return getMergedCoverageLength(relevant, start, end, gapToleranceM) / length;
}

function snapToNearestFixed(spans, fixedValue, toleranceM = 0.28) {
  const nearest = spans
    .map(span => ({ fixed: span.fixed, distance: Math.abs(span.fixed - fixedValue) }))
    .filter(item => item.distance <= toleranceM)
    .sort((a, b) => a.distance - b.distance)[0];
  return nearest ? nearest.fixed : fixedValue;
}

function clusterAlignedSpans(spans, fixedToleranceM = 0.18, gapToleranceM = 0.55) {
  const clusters = [];
  const sorted = [...spans].sort((a, b) => a.fixed - b.fixed || a.min - b.min);

  sorted.forEach(span => {
    let cluster = clusters.find(item => Math.abs(item.fixed - span.fixed) <= fixedToleranceM);
    if (!cluster) {
      cluster = {
        orientation: span.orientation,
        fixed: span.fixed,
        parts: []
      };
      clusters.push(cluster);
    }

    cluster.parts.push({ min: span.min, max: span.max });
    cluster.fixed = (cluster.fixed * (cluster.parts.length - 1) + span.fixed) / cluster.parts.length;
  });

  return clusters.flatMap(cluster => {
    const merged = [];
    const parts = cluster.parts
      .filter(part => part.max > part.min)
      .sort((a, b) => a.min - b.min);

    parts.forEach(part => {
      const last = merged[merged.length - 1];
      if (last && part.min <= last.max + gapToleranceM) {
        last.max = Math.max(last.max, part.max);
      } else {
        merged.push({ ...part });
      }
    });

    return merged.map(part => ({
      orientation: cluster.orientation,
      fixed: cluster.fixed,
      min: part.min,
      max: part.max
    }));
  }).sort((a, b) => a.fixed - b.fixed || a.min - b.min);
}

function rectArea(rect) {
  return Math.max(0, rect.maxX - rect.minX) * Math.max(0, rect.maxY - rect.minY);
}

function roomToRect(room) {
  if (!Array.isArray(room.polygon) || room.polygon.length < 3) return null;
  const xs = room.polygon.map(point => point.x);
  const ys = room.polygon.map(point => point.y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys)
  };
}

function rectOverlapInfo(a, b) {
  const overlapW = Math.max(0, Math.min(a.maxX, b.maxX) - Math.max(a.minX, b.minX));
  const overlapH = Math.max(0, Math.min(a.maxY, b.maxY) - Math.max(a.minY, b.minY));
  const overlapArea = overlapW * overlapH;
  const areaA = rectArea(a);
  const areaB = rectArea(b);
  return {
    overlapArea,
    ratioA: areaA > 0 ? overlapArea / areaA : 0,
    ratioB: areaB > 0 ? overlapArea / areaB : 0,
    ratioSmall: Math.min(areaA, areaB) > 0 ? overlapArea / Math.min(areaA, areaB) : 0
  };
}

function rectangleOverlapsRoom(rect, rooms, threshold = 0.18) {
  return rooms.some(room => {
    const existing = roomToRect(room);
    if (!existing) return false;
    const overlap = rectOverlapInfo(rect, existing);
    return overlap.overlapArea > 0.05 && overlap.ratioSmall > threshold;
  });
}

function polygonOverlapRatioApprox(polygonA, polygonB) {
  if (!Array.isArray(polygonA) || !Array.isArray(polygonB) || polygonA.length < 3 || polygonB.length < 3) {
    return 0;
  }
  const roomA = { polygon: polygonA };
  const roomB = { polygon: polygonB };
  const rectA = roomToRect(roomA);
  const rectB = roomToRect(roomB);
  const minX = Math.max(rectA.minX, rectB.minX);
  const maxX = Math.min(rectA.maxX, rectB.maxX);
  const minY = Math.max(rectA.minY, rectB.minY);
  const maxY = Math.min(rectA.maxY, rectB.maxY);
  if (maxX <= minX || maxY <= minY) return 0;

  const overlapWidth = maxX - minX;
  const overlapHeight = maxY - minY;
  const step = clamp(Math.min(overlapWidth, overlapHeight) / 10, 0.08, 0.35);
  let overlapSamples = 0;
  let totalSamples = 0;
  for (let y = minY + step / 2; y < maxY; y += step) {
    for (let x = minX + step / 2; x < maxX; x += step) {
      totalSamples++;
      if (pointInPolygon({ x, y }, polygonA) && pointInPolygon({ x, y }, polygonB)) {
        overlapSamples++;
      }
    }
  }
  if (!totalSamples) return 0;
  const overlapArea = overlapWidth * overlapHeight * overlapSamples / totalSamples;
  return overlapArea / Math.max(0.001, Math.min(polygonArea(polygonA), polygonArea(polygonB)));
}

function polygonOverlapsRoom(room, rooms, threshold = 0.5) {
  return rooms.some(existing =>
    polygonOverlapRatioApprox(room.polygon, existing.polygon) > threshold
  );
}

function getRoomCandidateScore(candidate) {
  const area = Number(candidate.areaM2 || 0);
  const width = Math.max(0.01, candidate.rect.maxX - candidate.rect.minX);
  const height = Math.max(0.01, candidate.rect.maxY - candidate.rect.minY);
  const aspect = Math.max(width / height, height / width);
  const areaBonus = Math.min(area, 24) * 0.035;
  const largePenalty = Math.max(0, area - 36) * 0.08;
  const shapePenalty = Math.max(0, aspect - 2.4) * 0.45;
  return candidate.coverageScore + areaBonus - largePenalty - shapePenalty;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

let blueprintAnalysisCache = null;

function getBlueprintAnalysisData() {
  const image = state.blueprintImgObj;
  if (!image?.width || !image?.height) return null;
  if (blueprintAnalysisCache?.image === image) return blueprintAnalysisCache;

  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(image, 0, 0);
  blueprintAnalysisCache = {
    image,
    width: image.width,
    height: image.height,
    pixels: context.getImageData(0, 0, image.width, image.height).data
  };
  return blueprintAnalysisCache;
}

function blueprintPixelBrightness(analysis, x, y) {
  if (!analysis || x < 0 || y < 0 || x >= analysis.width || y >= analysis.height) return 255;
  const index = (Math.floor(y) * analysis.width + Math.floor(x)) * 4;
  return (
    analysis.pixels[index] * 0.299
    + analysis.pixels[index + 1] * 0.587
    + analysis.pixels[index + 2] * 0.114
  );
}

function blueprintHasFilledWallStyle() {
  const analysis = getBlueprintAnalysisData();
  if (!analysis) return false;
  if (typeof analysis.hasFilledWallStyle === 'boolean') return analysis.hasFilledWallStyle;

  let dark = 0;
  let samples = 0;
  const step = 3;
  for (let y = 1; y < analysis.height - 1; y += step) {
    for (let x = 1; x < analysis.width - 1; x += step) {
      samples++;
      if (blueprintPixelBrightness(analysis, x, y) < 100) dark++;
    }
  }
  analysis.hasFilledWallStyle = dark / Math.max(1, samples) >= 0.052;
  return analysis.hasFilledWallStyle;
}

function blueprintHasDarkCoreNear(analysis, pixelX, pixelY, searchRadiusPx) {
  const radius = Math.max(2, Math.round(searchRadiusPx));
  const minX = clamp(Math.floor(pixelX - radius), 1, analysis.width - 2);
  const maxX = clamp(Math.ceil(pixelX + radius), 1, analysis.width - 2);
  const minY = clamp(Math.floor(pixelY - radius), 1, analysis.height - 2);
  const maxY = clamp(Math.ceil(pixelY + radius), 1, analysis.height - 2);

  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      let darkNeighbors = 0;
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          if (blueprintPixelBrightness(analysis, x + ox, y + oy) < 110) darkNeighbors++;
        }
      }
      if (darkNeighbors >= 7) return true;
    }
  }
  return false;
}

function blueprintThickWallSupport(wall) {
  const analysis = getBlueprintAnalysisData();
  const ppm = Number(state.pixelsPerMeter || 0);
  if (!analysis || ppm <= 0) return 0;

  const lengthM = Math.hypot(wall.p2x - wall.p1x, wall.p2y - wall.p1y);
  const sampleCount = Math.max(7, Math.ceil(lengthM / 0.18));
  const searchRadiusPx = clamp(ppm * 0.11, 3, 8);
  let hits = 0;

  for (let index = 0; index < sampleCount; index++) {
    const t = sampleCount === 1 ? 0.5 : index / (sampleCount - 1);
    const pixelX = (wall.p1x + (wall.p2x - wall.p1x) * t) * ppm;
    const pixelY = (wall.p1y + (wall.p2y - wall.p1y) * t) * ppm;
    if (blueprintHasDarkCoreNear(analysis, pixelX, pixelY, searchRadiusPx)) hits++;
  }
  return hits / sampleCount;
}

function markBlueprintWallCores(blocked, gridW, gridH, bounds, cellSize) {
  const analysis = getBlueprintAnalysisData();
  const ppm = Number(state.pixelsPerMeter || 0);
  if (!analysis || ppm <= 0) return;
  const searchRadiusPx = clamp(cellSize * ppm * 0.65, 3, 9);

  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const worldX = bounds.minX + (gx + 0.5) * cellSize;
      const worldY = bounds.minY + (gy + 0.5) * cellSize;
      if (blueprintHasDarkCoreNear(analysis, worldX * ppm, worldY * ppm, searchRadiusPx)) {
        blocked[gy * gridW + gx] = 1;
      }
    }
  }
}

function groupConsecutiveLineCenters(indices) {
  const groups = [];
  indices.forEach(index => {
    const last = groups.at(-1);
    if (!last || index > last.at(-1) + 1) groups.push([index]);
    else last.push(index);
  });
  return groups.map(group => group.reduce((sum, value) => sum + value, 0) / group.length);
}

function hasRepeatedLineSequence(centers, pixelsPerMeter) {
  if (centers.length < 6) return false;
  for (let start = 0; start <= centers.length - 6; start++) {
    const window = centers.slice(start, start + 6);
    const spacingsM = window.slice(1).map((center, index) =>
      (center - window[index]) / pixelsPerMeter
    );
    const typicalSpacing = median(spacingsM);
    if (typicalSpacing < 0.1 || typicalSpacing > 0.9) continue;
    if (spacingsM.every(spacing =>
      Math.abs(spacing - typicalSpacing) <= Math.max(0.14, typicalSpacing * 0.55)
    )) return true;
  }
  return false;
}

function roomHasBlueprintStairPattern(room) {
  const analysis = getBlueprintAnalysisData();
  const rect = roomToRect(room);
  const ppm = Number(state.pixelsPerMeter || 0);
  if (!analysis || !rect || ppm <= 0) return false;

  const x1 = clamp(Math.floor(rect.minX * ppm), 0, analysis.width - 1);
  const x2 = clamp(Math.ceil(rect.maxX * ppm), x1 + 1, analysis.width);
  const y1 = clamp(Math.floor(rect.minY * ppm), 0, analysis.height - 1);
  const y2 = clamp(Math.ceil(rect.maxY * ppm), y1 + 1, analysis.height);
  const width = x2 - x1;
  const height = y2 - y1;
  if (width < 24 || height < 24) return false;

  const marginX = Math.max(2, Math.floor(width * 0.08));
  const marginY = Math.max(2, Math.floor(height * 0.08));
  const innerWidth = width - marginX * 2;
  const innerHeight = height - marginY * 2;
  if (innerWidth < 16 || innerHeight < 16) return false;

  const horizontalLines = [];
  for (let localY = marginY; localY < height - marginY; localY++) {
    let darkPixels = 0;
    for (let localX = marginX; localX < width - marginX; localX++) {
      const pixelIndex = ((y1 + localY) * analysis.width + x1 + localX) * 4;
      const brightness = (
        analysis.pixels[pixelIndex] * 0.299
        + analysis.pixels[pixelIndex + 1] * 0.587
        + analysis.pixels[pixelIndex + 2] * 0.114
      );
      if (analysis.pixels[pixelIndex + 3] > 80 && brightness < 210) darkPixels++;
    }
    if (darkPixels >= innerWidth * 0.52) horizontalLines.push(localY);
  }

  const verticalLines = [];
  for (let localX = marginX; localX < width - marginX; localX++) {
    let darkPixels = 0;
    for (let localY = marginY; localY < height - marginY; localY++) {
      const pixelIndex = ((y1 + localY) * analysis.width + x1 + localX) * 4;
      const brightness = (
        analysis.pixels[pixelIndex] * 0.299
        + analysis.pixels[pixelIndex + 1] * 0.587
        + analysis.pixels[pixelIndex + 2] * 0.114
      );
      if (analysis.pixels[pixelIndex + 3] > 80 && brightness < 210) darkPixels++;
    }
    if (darkPixels >= innerHeight * 0.58) verticalLines.push(localX);
  }

  const horizontalCenters = groupConsecutiveLineCenters(horizontalLines);
  const verticalCenters = groupConsecutiveLineCenters(verticalLines);
  const horizontalPattern = horizontalCenters.length >= 7
    && hasRepeatedLineSequence(horizontalCenters, ppm);
  const verticalPattern = verticalCenters.length >= 9
    && hasRepeatedLineSequence(verticalCenters, ppm);
  return horizontalPattern || verticalPattern;
}

function roomHasStairPattern(room, walls) {
  const rect = roomToRect(room);
  if (!rect) return false;
  const roomWidth = rect.maxX - rect.minX;
  const roomHeight = rect.maxY - rect.minY;
  if (Math.min(roomWidth, roomHeight) < 1.2) return false;

  const edgeMargin = Math.min(0.28, Math.min(roomWidth, roomHeight) * 0.1);
  const spans = walls.map(normalizeWallSpan).filter(span => {
    if (!span) return false;
    const center = span.orientation === 'h'
      ? { x: (span.min + span.max) / 2, y: span.fixed }
      : { x: span.fixed, y: (span.min + span.max) / 2 };
    if (!pointInPolygon(center, room.polygon)) return false;

    const length = span.max - span.min;
    const crossSize = span.orientation === 'h' ? roomHeight : roomWidth;
    const alongSize = span.orientation === 'h' ? roomWidth : roomHeight;
    const nearBoundary = span.orientation === 'h'
      ? Math.min(Math.abs(span.fixed - rect.minY), Math.abs(span.fixed - rect.maxY)) < edgeMargin
      : Math.min(Math.abs(span.fixed - rect.minX), Math.abs(span.fixed - rect.maxX)) < edgeMargin;
    return !nearBoundary && length >= Math.max(0.45, alongSize * 0.28) && length <= alongSize * 1.08
      && crossSize >= 1.2;
  });

  const tracedWallPattern = ['h', 'v'].some(orientation => {
    const aligned = spans
      .filter(span => span.orientation === orientation)
      .sort((a, b) => a.fixed - b.fixed);
    if (aligned.length < 5) return false;

    for (let start = 0; start <= aligned.length - 5; start++) {
      const group = [aligned[start]];
      for (let index = start + 1; index < aligned.length; index++) {
        const previous = group[group.length - 1];
        const current = aligned[index];
        const spacing = current.fixed - previous.fixed;
        if (spacing > 0.9) break;

        const overlap = Math.max(0, Math.min(previous.max, current.max) - Math.max(previous.min, current.min));
        const shorter = Math.min(previous.max - previous.min, current.max - current.min);
        if (spacing >= 0.1 && overlap / Math.max(0.01, shorter) >= 0.62) {
          group.push(current);
        }
      }
      if (group.length < 5) continue;

      const spacings = group.slice(1).map((span, index) => span.fixed - group[index].fixed);
      const typicalSpacing = median(spacings);
      const consistentSpacing = spacings.filter(spacing =>
        spacing >= 0.1 && spacing <= 0.9 && Math.abs(spacing - typicalSpacing) <= Math.max(0.16, typicalSpacing * 0.55)
      ).length;
      const lengths = group.map(span => span.max - span.min);
      const typicalLength = median(lengths);
      const consistentLengths = lengths.filter(length =>
        Math.abs(length - typicalLength) <= Math.max(0.45, typicalLength * 0.4)
      ).length;
      if (consistentSpacing >= 4 && consistentLengths >= 5) return true;
    }
    return false;
  });
  return tracedWallPattern || roomHasBlueprintStairPattern(room);
}

function classifyDetectedRoomTypes(rooms, walls) {
  let stairNumber = 0;
  return rooms.map(room => {
    const areaM2 = Number(room.areaM2 || 0);
    if (areaM2 < 12 || areaM2 > 45 || !roomHasStairPattern(room, walls)) return room;
    stairNumber += 1;
    return {
      ...room,
      name: `Stairs ${stairNumber}`,
      type: 'stairs',
      roomType: 'stairs',
      priority: 0,
      clients: 0,
      coverageTarget: -80,
      excluded: true,
      detectedFeature: 'stairs'
    };
  });
}

function detectRectangularRoomsFromWalls(walls, existingRooms = []) {
  const spans = walls.map(normalizeWallSpan).filter(Boolean);
  const horizontal = clusterAlignedSpans(spans.filter(span => span.orientation === 'h'), 0.24, 1.25);
  const vertical = clusterAlignedSpans(spans.filter(span => span.orientation === 'v'), 0.18, 0.7);
  const candidates = [];
  const minAreaM2 = 1.2;
  const maxAreaM2 = 180;

  for (let topIndex = 0; topIndex < horizontal.length; topIndex++) {
    for (let bottomIndex = topIndex + 1; bottomIndex < horizontal.length; bottomIndex++) {
      const top = horizontal[topIndex];
      const bottom = horizontal[bottomIndex];
      const height = bottom.fixed - top.fixed;
      if (height < 0.8 || height > 14) continue;

      const verticalCandidates = vertical.filter(span =>
        Math.min(span.max, bottom.fixed) - Math.max(span.min, top.fixed) >= height * 0.36
      ).sort((a, b) => a.fixed - b.fixed);

      for (let leftIndex = 0; leftIndex < verticalCandidates.length - 1; leftIndex++) {
        for (let rightIndex = leftIndex + 1; rightIndex < verticalCandidates.length; rightIndex++) {
          const left = verticalCandidates[leftIndex];
          const right = verticalCandidates[rightIndex];
          const width = right.fixed - left.fixed;
          if (width < 0.8 || width > 10.5) continue;

          const horizontalTolerance = clamp(height * 0.12, 0.22, 0.55);
          const verticalTolerance = clamp(width * 0.08, 0.16, 0.4);
          const topCoverage = axisCoverageRatio(horizontal, top.fixed, left.fixed, right.fixed, horizontalTolerance, 1.25);
          const bottomCoverage = axisCoverageRatio(horizontal, bottom.fixed, left.fixed, right.fixed, horizontalTolerance, 1.25);
          const leftCoverage = axisCoverageRatio(vertical, left.fixed, top.fixed, bottom.fixed, verticalTolerance, 0.7);
          const rightCoverage = axisCoverageRatio(vertical, right.fixed, top.fixed, bottom.fixed, verticalTolerance, 0.7);
          const hasInternalDivider = verticalCandidates
            .slice(leftIndex + 1, rightIndex)
            .some(span => {
              const dividerMargin = Math.min(span.fixed - left.fixed, right.fixed - span.fixed);
              if (dividerMargin < 0.45) return false;
              return axisCoverageRatio(vertical, span.fixed, top.fixed, bottom.fixed, verticalTolerance, 0.45) >= 0.62;
            });
          const hasHorizontalDivider = horizontal
            .filter(span => span.fixed > top.fixed + 0.45 && span.fixed < bottom.fixed - 0.45)
            .some(span =>
              axisCoverageRatio(horizontal, span.fixed, left.fixed, right.fixed, horizontalTolerance, 0.55) >= 0.68
            );
          if (hasInternalDivider || hasHorizontalDivider) continue;

          const sideCoverages = [topCoverage, bottomCoverage, leftCoverage, rightCoverage];
          const strongSides = sideCoverages.filter(coverage => coverage >= 0.62).length;
          const usableSides = sideCoverages.filter(coverage => coverage >= 0.14).length;
          const totalCoverage = sideCoverages.reduce((sum, coverage) => sum + coverage, 0);
          const fourUsableSides = usableSides === 4 && strongSides >= 2 && totalCoverage >= 1.95;
          const threeStrongSides = strongSides >= 3
            && Math.min(...sideCoverages) >= 0.04
            && totalCoverage >= 1.95;
          if (!fourUsableSides && !threeStrongSides) continue;

          const areaM2 = width * height;
          if (areaM2 < minAreaM2 || areaM2 > maxAreaM2) continue;
          const aspectRatio = Math.max(width / height, height / width);
          if (aspectRatio > 5.5) continue;

          const rect = {
            minX: snapToNearestFixed(vertical, left.fixed, 0.25),
            maxX: snapToNearestFixed(vertical, right.fixed, 0.25),
            minY: snapToNearestFixed(horizontal, top.fixed, 0.3),
            maxY: snapToNearestFixed(horizontal, bottom.fixed, 0.3)
          };
          if (rect.maxX <= rect.minX || rect.maxY <= rect.minY) continue;
          if (rectangleOverlapsRoom(rect, existingRooms, 0.14)) continue;

          const polygon = [
            { x: rect.minX, y: rect.minY },
            { x: rect.maxX, y: rect.minY },
            { x: rect.maxX, y: rect.maxY },
            { x: rect.minX, y: rect.maxY }
          ];
          candidates.push({
            rect,
            coverageScore: totalCoverage,
            candidateScore: 0,
            id: `room_rect_${Date.now()}_${candidates.length + 1}`,
            name: `Room ${existingRooms.length + candidates.length + 1}`,
            type: 'room',
            roomType: 'office',
            areaM2,
            centroid: { x: (rect.minX + rect.maxX) / 2, y: (rect.minY + rect.maxY) / 2 },
            polygon,
            userDensity: 0.1,
            clients: estimateRoomClients(areaM2),
            priority: estimateRoomPriority(areaM2),
            coverageTarget: -67,
            excluded: false
          });
        }
      }
    }
  }

  const rooms = [];
  candidates
    .map(candidate => ({
      ...candidate,
      candidateScore: getRoomCandidateScore(candidate)
    }))
    .sort((a, b) => b.candidateScore - a.candidateScore || a.areaM2 - b.areaM2)
    .forEach(candidate => {
      if (!rectangleOverlapsRoom(candidate.rect, rooms, 0.16)) {
        rooms.push(candidate);
      }
    });

  return rooms;
}

function detectRoomsFromWalls() {
  const allWalls = getWallSegments();
  const filledWallStyle = blueprintHasFilledWallStyle();
  const walls = filledWallStyle
    ? allWalls.filter(wall => blueprintThickWallSupport(wall) >= 0.34)
    : allWalls;
  const doorOpenings = detectDoorOpenings(walls);
  const bounds = getWallDetectionBounds(walls);

  if (!bounds || walls.length < 3) {
    return [];
  }

  // Rectangular inference is useful for sparse line plans, but on filled-wall
  // plans it often mistakes furniture outlines for small partial rooms.
  const strictRectangularRooms = filledWallStyle
    ? []
    : detectRectangularRoomsFromWalls(walls, []);

  const longestSide = Math.max(bounds.widthM, bounds.heightM);
  const cellSize = clamp(longestSide / 260, 0.18, 0.45);
  const gridW = Math.max(8, Math.ceil(bounds.widthM / cellSize));
  const gridH = Math.max(8, Math.ceil(bounds.heightM / cellSize));
  const blocked = new Uint8Array(gridW * gridH);
  const wallRadiusM = Math.max(cellSize * 0.72, 0.12);

  walls.forEach(wall => {
    markBlockedSegment(blocked, gridW, gridH, bounds, cellSize, wall.p1x, wall.p1y, wall.p2x, wall.p2y, wallRadiusM);
  });
  if (filledWallStyle) {
    markBlueprintWallCores(blocked, gridW, gridH, bounds, cellSize);
  }
  const exteriorBoundary = detectBlueprintFootprintBoundary();
  exteriorBoundary?.forEach((point, index) => {
    const next = exteriorBoundary[(index + 1) % exteriorBoundary.length];
    markBlockedSegment(blocked, gridW, gridH, bounds, cellSize, point.x, point.y, next.x, next.y, wallRadiusM);
  });
  doorOpenings.forEach(opening => {
    markBlockedSegment(blocked, gridW, gridH, bounds, cellSize, opening.x1, opening.y1, opening.x2, opening.y2, wallRadiusM * 0.72);
  });

  const endpoints = getWallEndpointData(walls);
  const maxClosableGapM = Math.max(2.2, cellSize * 8);
  const maxCornerGapM = Math.max(0.65, cellSize * 2.5);
  for (let i = 0; i < endpoints.length; i++) {
    for (let j = i + 1; j < endpoints.length; j++) {
      const a = endpoints[i];
      const b = endpoints[j];
      const distance = Math.hypot(a.x - b.x, a.y - b.y);
      const directionSimilarity = Math.abs(a.dir.x * b.dir.x + a.dir.y * b.dir.y);
      const isDoorGap = distance > 0.12 && distance <= maxClosableGapM && directionSimilarity > 0.62;
      const isCornerGap = distance > 0.12 && distance <= maxCornerGapM;
      if (isDoorGap || isCornerGap) {
        markBlockedSegment(blocked, gridW, gridH, bounds, cellSize, a.x, a.y, b.x, b.y, wallRadiusM * 0.85);
      }
    }
  }

  const visited = new Uint8Array(gridW * gridH);
  const enclosedRooms = [];
  const minRoomAreaM2 = 1.2;
  const maxRoomAreaM2 = Math.min(bounds.widthM * bounds.heightM * 0.62, 180);
  const directions = [
    { x: 1, y: 0 },
    { x: -1, y: 0 },
    { x: 0, y: 1 },
    { x: 0, y: -1 }
  ];

  for (let sy = 0; sy < gridH; sy++) {
    for (let sx = 0; sx < gridW; sx++) {
      const startIndex = sy * gridW + sx;
      if (blocked[startIndex] || visited[startIndex]) continue;

      const stack = [{ x: sx, y: sy }];
      const component = [];
      let touchesBounds = false;
      let minCellX = sx;
      let maxCellX = sx;
      let minCellY = sy;
      let maxCellY = sy;
      visited[startIndex] = 1;

      while (stack.length) {
        const cell = stack.pop();
        component.push(cell);
        minCellX = Math.min(minCellX, cell.x);
        maxCellX = Math.max(maxCellX, cell.x);
        minCellY = Math.min(minCellY, cell.y);
        maxCellY = Math.max(maxCellY, cell.y);
        if (cell.x === 0 || cell.y === 0 || cell.x === gridW - 1 || cell.y === gridH - 1) {
          touchesBounds = true;
        }

        directions.forEach(direction => {
          const nx = cell.x + direction.x;
          const ny = cell.y + direction.y;
          if (nx < 0 || ny < 0 || nx >= gridW || ny >= gridH) return;
          const index = ny * gridW + nx;
          if (blocked[index] || visited[index]) return;
          visited[index] = 1;
          stack.push({ x: nx, y: ny });
        });
      }

      const areaM2 = component.length * cellSize * cellSize;
      const componentWidthM = (maxCellX - minCellX + 1) * cellSize;
      const componentHeightM = (maxCellY - minCellY + 1) * cellSize;
      const likelyOutsideAir =
        touchesBounds &&
        (areaM2 > 2.5 ||
          componentWidthM > bounds.widthM * 0.18 ||
          componentHeightM > bounds.heightM * 0.18);

      if (likelyOutsideAir || areaM2 < minRoomAreaM2 || areaM2 > maxRoomAreaM2) {
        continue;
      }

      const componentSet = new Set(component.map(cell => `${cell.x},${cell.y}`));
      const polygon = buildComponentPolygon(component, componentSet, bounds, cellSize, gridW, gridH);
      if (!polygon || !isLikelyEnclosedRoomPolygon(polygon)) continue;

      const refinedAreaM2 = polygonArea(polygon);
      if (refinedAreaM2 < minRoomAreaM2 || refinedAreaM2 > maxRoomAreaM2) {
        continue;
      }

      const centroid = polygonCentroid(polygon);
      enclosedRooms.push({
        id: `room_${Date.now()}_${enclosedRooms.length + 1}`,
        name: `Room ${enclosedRooms.length + 1}`,
        type: 'room',
        roomType: 'office',
        areaM2: refinedAreaM2,
        centroid,
        polygon,
        userDensity: 0.1,
        clients: estimateRoomClients(refinedAreaM2),
        priority: estimateRoomPriority(refinedAreaM2),
        coverageTarget: -67,
        excluded: false
      });
    }
  }

  // Prefer full flood-filled areas. Rectangle candidates only fill genuinely
  // missing zones and must not replace a larger enclosed room with a fragment.
  const mergedRooms = [...enclosedRooms]
    .sort((a, b) => Number(b.areaM2 || 0) - Number(a.areaM2 || 0));
  strictRectangularRooms
    .sort((a, b) => Number(b.areaM2 || 0) - Number(a.areaM2 || 0))
    .forEach(room => {
      if (!roomToRect(room) || polygonOverlapsRoom(room, mergedRooms, 0.2)) return;
      mergedRooms.push(room);
    });

  const sortedRooms = mergedRooms.sort((a, b) => a.centroid.y - b.centroid.y || a.centroid.x - b.centroid.x).map((room, index) => ({
    ...room,
    name: `Room ${index + 1}`
  }));
  return normalizeRooms(classifyDetectedRoomTypes(sortedRooms, walls));
}

function renderRoomsPanel() {
  const toggleButton = document.getElementById('btnToggleRoomsOverlay');
  if (toggleButton) toggleButton.textContent = state.showRoomsOverlay ? 'Hide Overlay' : 'Show Overlay';
  renderAutoPlacementPanel();
}

function openEditRoomModal(index) {
  const room = state.rooms?.[index];
  if (!room) return;
  state.selectedRoomIndex = index;
  renderRoomsPanel();
  draw();

  document.getElementById('editRoomName').value = room.name || `Room ${index + 1}`;
  document.getElementById('editRoomType').value = room.roomType || room.type || 'office';
  document.getElementById('editRoomPriority').value = String(room.priority ?? 1);
  document.getElementById('editRoomClients').value = Number(room.clients ?? 30);
  document.getElementById('editRoomCoverageTarget').value = Number(room.coverageTarget ?? -67);
  document.getElementById('editRoomArea').value = `${Number(room.areaM2 || 0).toFixed(1)} m²`;
  document.getElementById('editRoomExcluded').checked = Boolean(room.excluded);
  document.getElementById('editRoomModal').style.display = 'flex';
}

function closeEditRoomModal() {
  document.getElementById('editRoomModal').style.display = 'none';
}

document.getElementById('closeEditRoomModal')?.addEventListener('click', closeEditRoomModal);
document.getElementById('btnCancelEditRoom')?.addEventListener('click', closeEditRoomModal);

document.getElementById('btnSaveRoom')?.addEventListener('click', () => {
  const index = state.selectedRoomIndex;
  const room = state.rooms?.[index];
  if (!room) return;

  room.name = document.getElementById('editRoomName').value.trim() || room.name;
  room.roomType = document.getElementById('editRoomType').value;
  room.type = room.roomType;
  room.priority = Number(document.getElementById('editRoomPriority').value);
  room.clients = Math.max(0, Math.round(Number(document.getElementById('editRoomClients').value || 0)));
  room.userDensity = room.areaM2 > 0 ? room.clients / room.areaM2 : 0;
  room.coverageTarget = Number(document.getElementById('editRoomCoverageTarget').value || -67);
  room.excluded = document.getElementById('editRoomExcluded').checked;
  if (room.roomType === 'stairs') {
    room.priority = 0;
    room.clients = 0;
    room.userDensity = 0;
    room.coverageTarget = -80;
    room.excluded = true;
  }

  renderRoomsPanel();
  saveHistory();
  draw();
  closeEditRoomModal();
});

document.getElementById('btnDeleteRoom')?.addEventListener('click', async () => {
  const index = state.selectedRoomIndex;
  const room = state.rooms?.[index];
  if (!room) return;

  if (!(await customConfirm(`Delete ${room.name}?`, "Delete Room"))) return;
  state.rooms.splice(index, 1);
  state.selectedRoomIndex = null;
  renderRoomsPanel();
  saveHistory();
  draw();
  closeEditRoomModal();
});

document.querySelectorAll('.client-presets button').forEach(button => {
  button.addEventListener('click', () => {
    const input = document.getElementById('editRoomClients');
    if (input) input.value = button.dataset.clients;
  });
});

function drawRoomsOverlay(scalePx) {
  if (!state.showRoomsOverlay || !Array.isArray(state.rooms) || state.rooms.length === 0) return;

  state.rooms.forEach((room, index) => {
    if (!Array.isArray(room.polygon) || room.polygon.length < 3) return;

    ctx.beginPath();
    room.polygon.forEach((point, index) => {
      const x = point.x * scalePx;
      const y = point.y * scalePx;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    const selected = state.selectedRoomIndex === index;
    const roomFillAlpha = state.heatmapEnabled && state.accessPoints.length > 0 ? 0.07 : 0.18;
    ctx.fillStyle = room.excluded ? 'rgba(148, 163, 184, 0.14)' : `rgba(79, 70, 229, ${roomFillAlpha})`;
    ctx.strokeStyle = selected ? '#f59e0b' : (room.excluded ? '#94a3b8' : '#4338ca');
    ctx.lineWidth = selected ? 3 : 2;
    ctx.fill();
    ctx.stroke();

    if (room.areaM2 >= 2 && room.centroid) {
      const label = `${room.name || `Area ${index + 1}`} | ${Number(room.areaM2 || 0).toFixed(1)}m²`;
      ctx.font = '700 10px Inter, sans-serif';
      const labelWidth = ctx.measureText(label).width + 12;
      const labelHeight = 18;
      const labelX = room.centroid.x * scalePx;
      const labelY = room.centroid.y * scalePx;

      ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
      ctx.beginPath();
      ctx.roundRect(labelX - labelWidth / 2, labelY - labelHeight / 2, labelWidth, labelHeight, 5);
      ctx.fill();

      ctx.strokeStyle = 'rgba(67, 56, 202, 0.28)';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = '#3730a3';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, labelX, labelY);
    }
  });
}

function drawServiceBoundary(scalePx) {
  if (!Array.isArray(state.serviceBoundary) || state.serviceBoundary.length < 3) return;
  ctx.save();
  ctx.beginPath();
  state.serviceBoundary.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x * scalePx, point.y * scalePx);
    else ctx.lineTo(point.x * scalePx, point.y * scalePx);
  });
  ctx.closePath();
  ctx.setLineDash([8, 5]);
  ctx.strokeStyle = '#059669';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function drawDoorOpenings(scalePx) {
  if (!state.showRoomsOverlay || !Array.isArray(state.doorOpenings)) return;
  ctx.save();
  ctx.setLineDash([3, 3]);
  ctx.strokeStyle = '#0891b2';
  ctx.lineWidth = 3;
  state.doorOpenings.forEach(opening => {
    ctx.beginPath();
    ctx.moveTo(opening.x1 * scalePx, opening.y1 * scalePx);
    ctx.lineTo(opening.x2 * scalePx, opening.y2 * scalePx);
    ctx.stroke();
  });
  ctx.restore();
}

function drawInteractiveHeatmapPreview(scalePx) {
  const bounds = getHeatmapBounds();
  if (!bounds) return;

  const heatmapX = bounds.minX * scalePx;
  const heatmapY = bounds.minY * scalePx;
  const heatmapW = bounds.widthM * scalePx;
  const heatmapH = bounds.heightM * scalePx;

  ctx.save();
  ctx.beginPath();
  ctx.rect(heatmapX, heatmapY, heatmapW, heatmapH);
  ctx.clip();

  state.accessPoints.forEach(ap => {
    const px = ap.x * scalePx;
    const py = ap.y * scalePx;
    const radiusM = estimateSignalRadiusM(ap, -88);
    const radiusPx = radiusM * scalePx;
    const strongStop = clamp(estimateSignalRadiusM(ap, -58) / radiusM, 0.04, 0.22);
    const goodStop = clamp(estimateSignalRadiusM(ap, -66) / radiusM, strongStop + 0.04, 0.38);
    const midStop = clamp(estimateSignalRadiusM(ap, -74) / radiusM, goodStop + 0.04, 0.62);

    const gradient = ctx.createRadialGradient(px, py, 0, px, py, radiusPx);
    gradient.addColorStop(0, 'rgba(148, 218, 92, 0.82)');
    gradient.addColorStop(strongStop, 'rgba(155, 219, 92, 0.76)');
    gradient.addColorStop(goodStop, 'rgba(194, 226, 81, 0.67)');
    gradient.addColorStop(midStop, 'rgba(255, 232, 84, 0.55)');
    gradient.addColorStop(0.82, 'rgba(255, 186, 107, 0.34)');
    gradient.addColorStop(1, 'rgba(255, 153, 170, 0)');

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(px, py, radiusPx, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.restore();
}

function updateLegendHover(pos) {
  const legend = document.querySelector('.legend');
  const marker = document.getElementById('legendHoverMarker');
  const markerValue = document.getElementById('legendHoverValue');
  const topLabel = document.getElementById('legendTopLabel');
  const bottomLabel = document.getElementById('legendBottomLabel');

  if (!legend || !marker || !markerValue || state.accessPoints.length === 0 || !state.heatmapEnabled) {
    resetLegendHover();
    return;
  }

  const bounds = getHeatmapBounds();
  if (!bounds) {
    resetLegendHover();
    return;
  }

  const pxM = pos.x / state.pixelsPerMeter;
  const pyM = pos.y / state.pixelsPerMeter;
  const isInsideHeatmap =
    pxM >= bounds.minX &&
    pxM <= bounds.minX + bounds.widthM &&
    pyM >= bounds.minY &&
    pyM <= bounds.minY + bounds.heightM;

  if (!isInsideHeatmap) {
    resetLegendHover();
    return;
  }

  const walls = getWallSegments();
  let bestSignal = -999;
  let bestAp = null;
  for (const ap of state.accessPoints) {
    const signal = calcSignalAt(pxM, pyM, ap, walls);
    if (signal > bestSignal) {
      bestSignal = signal;
      bestAp = ap;
    }
  }

  const range = getLegendSignalRange();
  const clampedSignal = clamp(bestSignal, range.poor, range.good);
  const markerTop = ((range.good - clampedSignal) / (range.good - range.poor)) * 100;
  const quality = getSignalQualityLabel(bestSignal);
  const displaySignal = Math.round(bestSignal);
  const wallTrace = bestAp
    ? traceWallAttenuation(bestAp.x, bestAp.y, pxM, pyM, walls, getActiveBandKey())
    : { count: 0, totalDb: 0 };

  legend.classList.add('is-hovering');
  marker.style.display = 'flex';
  marker.style.top = `${markerTop}%`;
  markerValue.textContent = `${displaySignal} dBm · ${bestAp?.name || 'AP'} · ${wallTrace.count} wall / ${Math.round(wallTrace.totalDb)} dB`;
  if (topLabel) topLabel.textContent = quality;
  if (bottomLabel) bottomLabel.innerHTML = `${displaySignal} dBm<br><small>${getActiveFrequencyLabel()}</small>`;
}

/**
 * Compute the full heatmap and cache it as an offscreen canvas.
 * Uses a lower resolution grid for performance.
 */
function computeHeatmap() {
  if (state.accessPoints.length === 0) {
    heatmapCache = null;
    heatmapWorldBounds = null;
    return;
  }

  const ppm = state.pixelsPerMeter;
  const bounds = getHeatmapBounds();

  if (!bounds) {
    heatmapCache = null;
    heatmapWorldBounds = null;
    return;
  }

  const heatmapWidthPx = Math.max(1, Math.ceil(bounds.widthM * ppm));
  const heatmapHeightPx = Math.max(1, Math.ceil(bounds.heightM * ppm));
  const isInteractiveMove = isHeatmapInteractiveMove();
  const targetGridSize = isInteractiveMove ? 150 : 260;
  const minStep = isInteractiveMove ? 7 : 4;
  const maxStep = isInteractiveMove ? 14 : 8;
  const step = Math.max(
    minStep,
    Math.min(maxStep, Math.round(Math.max(heatmapWidthPx, heatmapHeightPx) / targetGridSize))
  );
  const gridW = Math.ceil(heatmapWidthPx / step);
  const gridH = Math.ceil(heatmapHeightPx / step);

  // Prepare wall segments in meter coordinates
  const walls = getWallSegments();

  // Create offscreen canvas
  const offscreen = document.createElement('canvas');
  offscreen.width = gridW;
  offscreen.height = gridH;
  const offCtx = offscreen.getContext('2d');
  const imageData = offCtx.createImageData(gridW, gridH);
  const data = imageData.data;

  // Compute signal at each grid point
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      // Convert grid position to meters
      const px = bounds.minX + ((gx + 0.5) * step) / ppm;
      const py = bounds.minY + ((gy + 0.5) * step) / ppm;

      // Get best signal from all APs (strongest signal wins)
      let bestSignal = -999;
      const simulationProfile = getSimulationProfile(getActiveFrequencyGHz());
      for (const ap of state.accessPoints) {
        const sig = calcSignalAt(px, py, ap, walls);
        if (sig > bestSignal) bestSignal = sig;
      }

      if (walls.length > 0 && simulationProfile.wallPenaltyRadiusM > 0) {
        let nearestWallM = Infinity;
        for (const wall of walls) {
          const wallDistance = distancePointToSegment(px, py, wall.p1x, wall.p1y, wall.p2x, wall.p2y);
          nearestWallM = Math.min(nearestWallM, wallDistance);
        }

        if (nearestWallM < simulationProfile.wallPenaltyRadiusM) {
          bestSignal -= lerp(
            simulationProfile.wallPenaltyDb,
            0.8,
            nearestWallM / simulationProfile.wallPenaltyRadiusM
          );
        }
      }

      // Map signal to color
      const color = signalToColor(bestSignal);
      const idx = (gy * gridW + gx) * 4;
      data[idx] = color[0]; // R
      data[idx + 1] = color[1]; // G
      data[idx + 2] = color[2]; // B
      data[idx + 3] = color[3]; // A
    }
  }

  offCtx.putImageData(imageData, 0, 0);

  const smoothed = document.createElement('canvas');
  smoothed.width = heatmapWidthPx;
  smoothed.height = heatmapHeightPx;
  const smoothCtx = smoothed.getContext('2d');

  if ('filter' in smoothCtx) {
    smoothCtx.filter = 'blur(8px) saturate(108%)';
  }
  smoothCtx.imageSmoothingEnabled = true;
  smoothCtx.drawImage(offscreen, 0, 0, heatmapWidthPx, heatmapHeightPx);
  if ('filter' in smoothCtx) {
    smoothCtx.filter = 'none';
  }
  smoothCtx.globalAlpha = 0.42;
  smoothCtx.drawImage(offscreen, 0, 0, heatmapWidthPx, heatmapHeightPx);
  smoothCtx.globalAlpha = 1;

  heatmapCache = smoothed;
  heatmapWorldBounds = bounds;
  heatmapDirty = false;
}

// Add heatmap toggle to state
state.heatmapEnabled = true;

// Drawing logic
function draw() {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Disable image smoothing for sharp blueprint lines
  ctx.imageSmoothingEnabled = false;
  ctx.webkitImageSmoothingEnabled = false;
  ctx.mozImageSmoothingEnabled = false;

  ctx.translate(state.transform.x, state.transform.y);
  ctx.scale(state.transform.scale, state.transform.scale);

  // Update heatmap image position if it exists
  if (state.heatmapImageSrc && state.heatmapBounds) {
    const scalePx = state.pixelsPerMeter;
    const minX = state.heatmapBounds.min_x * scalePx;
    const minY = state.heatmapBounds.min_y * scalePx;
    const width = (state.heatmapBounds.max_x - state.heatmapBounds.min_x) * scalePx;
    const height = (state.heatmapBounds.max_y - state.heatmapBounds.min_y) * scalePx;

    heatmapImage.style.display = 'block';
    heatmapImage.src = state.heatmapImageSrc;

    // Apply transform to image container
    const absX = state.transform.x + (minX * state.transform.scale);
    const absY = state.transform.y + (minY * state.transform.scale);

    heatmapImage.style.transform = `translate(${absX}px, ${absY}px) scale(${state.transform.scale})`;
    heatmapImage.style.width = `${width}px`;
    heatmapImage.style.height = `${height}px`;
  } else {
    heatmapImage.style.display = 'none';
  }

  const scalePx = state.pixelsPerMeter; // 1m = state.pixelsPerMeterpx
  const scaleText = document.getElementById("currentScaleText");
  if (scaleText) scaleText.innerText = `1m: ${state.pixelsPerMeter.toFixed(1)}px`;

  // Draw blueprint background
  if (state.blueprintImgObj) {
    ctx.globalAlpha = state.heatmapEnabled ? 0.68 : 0.8;
    ctx.drawImage(state.blueprintImgObj, 0, 0);
    ctx.globalAlpha = 1.0;
  }

  // Draw WiFi heatmap overlay
  if (state.heatmapEnabled && state.accessPoints.length > 0) {
    const interactiveHeatmap = isHeatmapInteractiveMove();
    if (interactiveHeatmap) {
      drawInteractiveHeatmapPreview(scalePx);
    } else if (heatmapDirty || !heatmapCache) {
      computeHeatmap();
    }
    if (!interactiveHeatmap && heatmapCache && heatmapWorldBounds) {
      const heatmapX = heatmapWorldBounds.minX * scalePx;
      const heatmapY = heatmapWorldBounds.minY * scalePx;
      const heatmapW = heatmapWorldBounds.widthM * scalePx;
      const heatmapH = heatmapWorldBounds.heightM * scalePx;

      ctx.globalAlpha = 0.78;
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(heatmapCache, heatmapX, heatmapY, heatmapW, heatmapH);
      ctx.imageSmoothingEnabled = false;
      ctx.globalAlpha = 1.0;
    }
  }

  // Draw walls
  state.elements.forEach((elem, index) => {
    if (elem.type === 'wall' && elem.points.length === 2) {
      const wallType = state.wallTypes.find(w => w.id === elem.material) || state.wallTypes[0];
      const p1 = { x: elem.points[0][0] * scalePx, y: elem.points[0][1] * scalePx };
      const p2 = { x: elem.points[1][0] * scalePx, y: elem.points[1][1] * scalePx };

      ctx.lineWidth = Math.max(2, (wallType.thickness / 50) * 1.5);
      ctx.strokeStyle = wallType.color;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      // Highlight if selected
      if (state.selectedElementType === 'wall' && state.selectedElementIndex === index) {
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw handles if hovered or being dragged
      const isHovered = (state.hoveredHandle && state.hoveredHandle.elementIndex === index) || state.hoveredWall === index;
      const isDragged = state.draggingHandle && state.draggingHandle.elementIndex === index;

      if (isHovered || isDragged) {
        [p1, p2].forEach((p, pIdx) => {
          // Outer border matching wall color
          ctx.beginPath();
          ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
          ctx.fillStyle = wallType.color;
          ctx.fill();

          // Inner white
          ctx.beginPath();
          ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = '#fff';
          ctx.fill();

          // If this specific point is hovered/dragged, add a small center dot
          const isThisPoint = (state.hoveredHandle?.pointIndex === pIdx && state.hoveredHandle?.elementIndex === index) ||
            (state.draggingHandle?.pointIndex === pIdx && state.draggingHandle?.elementIndex === index);
          if (isThisPoint) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
            ctx.fillStyle = wallType.color;
            ctx.fill();
          }
        });

        // Draw pill label with info
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        const lengthM = Math.sqrt(dx * dx + dy * dy) / scalePx;

        const label = `${wallType.name} ${lengthM.toFixed(2)}m`;
        ctx.font = '600 11px Inter, sans-serif';
        const textWidth = ctx.measureText(label).width;

        // Background pill
        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
        ctx.beginPath();
        // ctx.roundRect is not always available, use simple rect for safety or polyfill
        const r = 10;
        const labelY = midY - 25;
        const x = midX - textWidth / 2 - 8;
        const y = labelY - 10;
        const w = textWidth + 16;
        const h = 20;
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = wallType.color;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Text
        ctx.fillStyle = '#1e293b';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, midX, labelY);
      }
    }
  });

  drawServiceBoundary(scalePx);
  drawDoorOpenings(scalePx);
  drawRoomsOverlay(scalePx);

  // Draw current line
  if (state.currentLine) {
    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(state.currentLine[0].x, state.currentLine[0].y);
    ctx.lineTo(state.currentLine[1].x, state.currentLine[1].y);
    ctx.stroke();

    // Draw endpoint points
    ctx.fillStyle = '#3b82f6';
    [state.currentLine[0], state.currentLine[1]].forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.lineWidth = 1; // Reset
  }

  // Draw APs
  state.accessPoints.forEach(ap => {
    const px = ap.x * scalePx;
    const py = ap.y * scalePx;
    const modelId = ap.model_id || 'u6_pro';
    const img = apImageCache[modelId];

    if (img) {
      const radius = 16 * state.iconSize;
      const size = 26 * state.iconSize;

      // Draw circular background
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.fillStyle = '#334155';
      ctx.fill();

      // Draw actual AP image
      ctx.save();

      // Draw circular clip
      ctx.beginPath();
      ctx.arc(px, py, size / 2, 0, Math.PI * 2);
      ctx.clip();

      ctx.drawImage(img, px - size / 2, py - size / 2, size, size);
      ctx.restore();

      // Ring border
      ctx.beginPath();
      ctx.arc(px, py, radius, 0, Math.PI * 2);
      ctx.strokeStyle = '#94a3b8';
      ctx.lineWidth = 1;
      ctx.stroke();
    } else {
      // Fallback to circle if image not loaded
      ctx.beginPath();
      ctx.arc(px, py, 14, 0, Math.PI * 2);
      ctx.fillStyle = '#334155';
      ctx.fill();
    }

    // Name tag
    ctx.fillStyle = '#475569';
    ctx.beginPath();
    ctx.roundRect(px - 15, py + 18, 30, 12, 4);
    ctx.fill();

    ctx.fillStyle = 'white';
    ctx.font = '7px Inter';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(ap.name, px, py + 24);
  });

  ctx.restore();

  updateFloatingToolbar();
}

// Heatmap toggle
document.getElementById('btnToggleHeatmap')?.addEventListener('click', (e) => {
  e.preventDefault();
  setHeatmapEnabled(!state.heatmapEnabled);
  draw();
});

function getProjectExportName() {
  return document.getElementById('topBarProjectName')?.innerText || "WiFi Project";
}

function buildProjectPayload(name = getProjectExportName()) {
  return {
    name,
    elements: state.elements,
    access_points: state.accessPoints,
    parameters: {
      name,
      pixelsPerMeter: state.pixelsPerMeter,
      blueprintImageSrc: state.blueprintImageSrc,
      floorHeight: state.floorHeight,
      selectedFrequency: getActiveFrequencyGHz(),
      selectedApModel: state.selectedApModel,
      selectedWallType: state.selectedWallType,
      wallTypes: state.wallTypes,
      apModels: state.apModels,
      rooms: state.rooms || [],
      serviceBoundary: state.serviceBoundary || [],
      doorOpenings: state.doorOpenings || [],
      showRoomsOverlay: state.showRoomsOverlay,
      autoPlacementLocked: state.autoPlacementLocked,
      apPlacementTab: state.apPlacementTab,
      rlAgentPlacement: (() => {
        const el = document.getElementById('chkRlAgentPlacement');
        return el ? !!el.checked : true; // default: RL agent only (no auto-added APs)
      })(),
      manualAccessPointsBackup: state.manualAccessPointsBackup || [],
      autoAccessPointsBackup: state.autoAccessPointsBackup || []
    },
    export_date: new Date().toISOString(),
    version: "2.0"
  };
}

function downloadJsonFile(data, filename) {
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(data, null, 2));
  const downloadAnchorNode = document.createElement('a');
  downloadAnchorNode.setAttribute("href", dataStr);
  downloadAnchorNode.setAttribute("download", filename);
  document.body.appendChild(downloadAnchorNode);
  downloadAnchorNode.click();
  downloadAnchorNode.remove();
}

document.getElementById('btnSave')?.addEventListener('click', async () => {
  try {
    const btn = document.getElementById('btnSave');
    const oldText = btn.innerText;

    if (!projectId) {
      const name = await customPrompt("No project loaded. Enter a name to create a new project:", "My WiFi Plan", "Create Project");
      if (!name) return;

      btn.innerText = "Creating...";
      btn.disabled = true;

      // Create new project entry
      const createRes = await fetch(`${API_BASE}/projects`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name })
      });

      if (!createRes.ok) throw new Error("Failed to create project");
      const project = await createRes.json();
      projectId = project.id;

      // Update URL without reload
      const newUrl = window.location.protocol + "//" + window.location.host + window.location.pathname + '?id=' + projectId;
      window.history.pushState({ path: newUrl }, '', newUrl);

      const topBarName = document.getElementById('topBarProjectName');
      if (topBarName) topBarName.innerText = name;

      btn.disabled = false;
    }

    btn.innerText = "Saving...";
    btn.disabled = true;

    const res = await fetch(`${API_BASE}/projects/${projectId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildProjectPayload(getProjectExportName()))
    });

    const data = await res.json();
    if (data.success) {
      btn.innerText = "Saved!";
      btn.style.backgroundColor = "#059669"; // Success green
      setTimeout(() => {
        btn.innerText = oldText;
        btn.style.backgroundColor = "#10b981"; // Original emerald
        btn.disabled = false;
      }, 2000);
    } else {
      throw new Error(data.error || "Failed to save");
    }
  } catch (err) {
    console.error(err);
    await customAlert('Failed to save project: ' + err.message);
    const btn = document.getElementById('btnSave');
    if (btn) {
      btn.innerText = "Save";
      btn.disabled = false;
    }
  }
});

document.getElementById('btnExport')?.addEventListener('click', () => {
  try {
    const projectData = buildProjectPayload();
    downloadJsonFile(projectData, `WiFi_Planner_${projectData.name.replace(/\s+/g, '_')}.json`);
  } catch (err) {
    console.error("Export error:", err);
    customAlert("Failed to export project.");
  }
});

document.getElementById('btnExportRl')?.addEventListener('click', async () => {
  try {
    const btn = document.getElementById('btnExportRl');
    const oldText = btn.innerText;
    btn.innerText = "Preparing...";
    btn.disabled = true;

    const projectData = buildProjectPayload();
    const resp = await fetch(`${API_BASE}/rl/scenario`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(projectData)
    });

    if (!resp.ok) throw new Error(`RL scenario export failed with ${resp.status}`);
    const scenario = await resp.json();
    downloadJsonFile(scenario, `RL_Scenario_${projectData.name.replace(/\s+/g, '_')}.json`);

    btn.innerText = oldText;
    btn.disabled = false;
  } catch (err) {
    console.error("RL export error:", err);
    const btn = document.getElementById('btnExportRl');
    if (btn) {
      btn.innerText = "RL Export";
      btn.disabled = false;
    }
    customAlert("Failed to export RL scenario. Make sure the backend is running.");
  }
});

// Device List Modal Logic
document.getElementById('btnDeviceList')?.addEventListener('click', (e) => {
  e.preventDefault();
  renderDeviceList();
  document.getElementById('deviceListModal').style.display = 'flex';
});

document.getElementById('closeDeviceList')?.addEventListener('click', () => {
  document.getElementById('deviceListModal').style.display = 'none';
});

function renderDeviceList() {
  const tbody = document.getElementById('apTableBody');
  if (!tbody) return;
  tbody.innerHTML = '';
  const activeFrequency = getActiveFrequencyGHz();
  const activeBandLabel = getActiveFrequencyLabel();
  const activeGainLabel = activeFrequency >= 5 ? 'Antenna Gain (5 GHz)' : 'Antenna Gain (2.4 GHz)';
  const powerHeader = document.querySelector('#apTable thead th:nth-child(4)');
  const gainHeader = document.querySelector('#apTable thead th:nth-child(5)');
  const channelHeader = document.querySelector('#apTable thead th:nth-child(6)');
  if (powerHeader) powerHeader.textContent = `TX Power (${activeBandLabel})`;
  if (gainHeader) gainHeader.textContent = activeGainLabel;
  if (channelHeader) channelHeader.textContent = `Channel (${activeBandLabel})`;

  state.accessPoints.forEach((ap, index) => {
    const apModel = getApModelById(ap.model_id || state.selectedApModel);
    const activeGain = getAntennaGainForBand(apModel, ap, activeFrequency);
    const activePower = getTxPowerForBand(apModel, ap, activeFrequency);
    const activeChannel = activeFrequency >= 5 ? ap.channel5 : ap.channel24;
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>
        <strong>${ap.name}</strong><br>
        <small>${ap.vendor} ${ap.model || ''}</small>
      </td>
      <td>${ap.x.toFixed(2)}</td>
      <td>${ap.y.toFixed(2)}</td>
      <td>${activePower.toFixed(1)}</td>
      <td>${activeGain} dBi</td>
      <td>${activeChannel || '-'}</td>
      <td>
        <button onclick="editAPPower(${index})" style="color: #2563eb; background: none; border: none; cursor: pointer; font-size: 0.8rem; margin-right: 10px;">Power</button>
        <button onclick="deleteAP(${index})" style="color: #ef4444; background: none; border: none; cursor: pointer; font-size: 0.8rem;">Delete</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

window.editAPPower = async (index) => {
  const ap = state.accessPoints[index];
  const apModel = getApModelById(ap.model_id || state.selectedApModel);
  const activeFrequency = getActiveFrequencyGHz();
  const currentPower = getTxPowerForBand(apModel, ap, activeFrequency);
  const defaultPower = getDefaultTxPowerForBand(apModel, activeFrequency);
  const newValue = await customPrompt(
    `Enter TX power for ${ap.name} on ${getActiveFrequencyLabel()} (dBm):`,
    String(currentPower),
    `Adjust ${ap.name}`
  );

  if (!newValue || Number.isNaN(Number(newValue))) {
    return;
  }

  const nextPower = clamp(Number(newValue), 1, Number(apModel.maxPower || 30));
  if (activeFrequency >= 5) {
    ap.power5 = nextPower;
  } else {
    ap.power24 = nextPower;
  }
  ap.power = nextPower;
  ap.frequency = activeFrequency;

  if (Math.abs(nextPower - defaultPower) > 0.01) {
    ap.customPower = true;
  }

  state.accessPoints[index] = normalizeAccessPoint(ap);
  invalidateHeatmap();
  saveHistory();
  draw();
  renderDeviceList();
};

window.deleteAP = async (index) => {
  if (await customConfirm(`Delete ${state.accessPoints[index].name}?`)) {
    state.accessPoints.splice(index, 1);
    saveHistory();
    draw();
    renderDeviceList();
  }
};

// Workspace Dropzone Logic
const workspaceDropzone = document.getElementById('workspaceDropzone');
const workspaceFileInput = document.getElementById('workspaceFileInput');

if (workspaceDropzone) {
  workspaceDropzone.addEventListener('click', () => workspaceFileInput.click());

  workspaceDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    workspaceDropzone.classList.add('dragover');
  });

  workspaceDropzone.addEventListener('dragleave', () => {
    workspaceDropzone.classList.remove('dragover');
  });

  workspaceDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    workspaceDropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      handleWorkspaceFile(e.dataTransfer.files[0]);
    }
  });

  workspaceFileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
      handleWorkspaceFile(e.target.files[0]);
    }
  });
}

async function handleWorkspaceFile(file) {
  if (!file.type.startsWith('image/')) {
    await customAlert("Please upload an image file.");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    state.blueprintImageSrc = e.target.result;
    const img = new Image();
    img.onload = () => {
      state.blueprintImgObj = img;
      document.getElementById('workspaceEmptyState').style.display = 'none';
      draw();
    };
    img.src = state.blueprintImageSrc;
  };
  reader.readAsDataURL(file);
}

// Custom Dialogs
function customAlert(message, title = "Notification") {
  return new Promise((resolve) => {
    const modal = document.getElementById('genericPromptModal');
    const titleEl = document.getElementById('promptTitle');
    const msgEl = document.getElementById('promptMessage');
    const inputEl = document.getElementById('promptInput');
    const btnOk = document.getElementById('btnPromptOk');
    const btnCancel = document.getElementById('btnPromptCancel');

    titleEl.innerText = title;
    msgEl.innerText = message;
    inputEl.style.display = 'none';
    btnCancel.style.display = 'none';
    modal.style.display = 'flex';

    const cleanup = () => {
      modal.style.display = 'none';
      btnOk.removeEventListener('click', onOk);
      document.getElementById('closePrompt').removeEventListener('click', cleanup);
    };

    const onOk = () => {
      cleanup();
      resolve();
    };

    btnOk.addEventListener('click', onOk);
    document.getElementById('closePrompt').addEventListener('click', cleanup);
  });
}

function customPrompt(message, defaultValue = "", title = "Input Required") {
  return new Promise((resolve) => {
    const modal = document.getElementById('genericPromptModal');
    const titleEl = document.getElementById('promptTitle');
    const msgEl = document.getElementById('promptMessage');
    const inputEl = document.getElementById('promptInput');
    const btnOk = document.getElementById('btnPromptOk');
    const btnCancel = document.getElementById('btnPromptCancel');

    titleEl.innerText = title;
    msgEl.innerText = message;
    inputEl.style.display = 'block';
    inputEl.value = defaultValue;
    btnCancel.style.display = 'block';
    modal.style.display = 'flex';
    inputEl.focus();

    const cleanup = () => {
      modal.style.display = 'none';
      btnOk.removeEventListener('click', onOk);
      btnCancel.removeEventListener('click', onCancel);
      document.getElementById('closePrompt').removeEventListener('click', onCancel);
    };

    const onOk = () => {
      const val = inputEl.value;
      cleanup();
      resolve(val);
    };

    const onCancel = () => {
      cleanup();
      resolve(null);
    };

    btnOk.addEventListener('click', onOk);
    btnCancel.addEventListener('click', onCancel);
    document.getElementById('closePrompt').addEventListener('click', onCancel);

    // Support Enter key
    inputEl.onkeydown = (e) => {
      if (e.key === 'Enter') onOk();
      if (e.key === 'Escape') onCancel();
    };
  });
}

function customConfirm(message, title = "Confirm Action") {
  return new Promise((resolve) => {
    const modal = document.getElementById('genericPromptModal');
    const titleEl = document.getElementById('promptTitle');
    const msgEl = document.getElementById('promptMessage');
    const inputEl = document.getElementById('promptInput');
    const btnOk = document.getElementById('btnPromptOk');
    const btnCancel = document.getElementById('btnPromptCancel');

    titleEl.innerText = title;
    msgEl.innerText = message;
    inputEl.style.display = 'none';
    btnCancel.style.display = 'block';
    modal.style.display = 'flex';

    const cleanup = () => {
      modal.style.display = 'none';
      btnOk.removeEventListener('click', onOk);
      btnCancel.removeEventListener('click', onCancel);
      document.getElementById('closePrompt').removeEventListener('click', onCancel);
    };

    const onOk = () => {
      cleanup();
      resolve(true);
    };

    const onCancel = () => {
      cleanup();
      resolve(false);
    };

    btnOk.addEventListener('click', onOk);
    btnCancel.addEventListener('click', onCancel);
    document.getElementById('closePrompt').addEventListener('click', onCancel);
  });
}

// Draggable panels
function makeDraggable(element, handle) {
  let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
  handle.style.cursor = 'move';
  handle.onmousedown = dragMouseDown;

  function dragMouseDown(e) {
    e = e || window.event;
    e.preventDefault();
    pos3 = e.clientX;
    pos4 = e.clientY;
    document.onmouseup = closeDragElement;
    document.onmousemove = elementDrag;
  }

  function elementDrag(e) {
    e = e || window.event;
    e.preventDefault();
    pos1 = pos3 - e.clientX;
    pos2 = pos4 - e.clientY;
    pos3 = e.clientX;
    pos4 = e.clientY;
    element.style.top = (element.offsetTop - pos2) + "px";
    element.style.left = (element.offsetLeft - pos1) + "px";
    // Disable right/bottom constraints
    element.style.right = 'auto';
    element.style.bottom = 'auto';
  }

  function closeDragElement() {
    document.onmouseup = null;
    document.onmousemove = null;
  }
}

// Initialize draggables
const wallPanel = document.getElementById('drawWallsPanel');
if (wallPanel) makeDraggable(wallPanel, wallPanel.querySelector('.panel-header'));

const devicePanel = document.getElementById('placeDevicesPanel');
if (devicePanel) makeDraggable(devicePanel, devicePanel.querySelector('.panel-header'));

const genericModal = document.querySelector('#genericPromptModal .modal-content');
if (genericModal) makeDraggable(genericModal, genericModal.querySelector('.modal-header'));

// Floating toolbar listeners
document.getElementById('btnCopyWall')?.addEventListener('click', (e) => {
  e.stopPropagation();
  copySelectedElement();
});

document.getElementById('btnReplaceWall')?.addEventListener('click', (e) => {
  e.stopPropagation();
  const wallPanel = document.getElementById('drawWallsPanel');
  if (wallPanel) {
    wallPanel.style.display = 'flex';
    renderWallPanel();
  }
});

document.getElementById('btnDeleteWall')?.addEventListener('click', (e) => {
  e.stopPropagation();
  deleteSelectedElement();
});

function findHandleAtPos(pos) {
  const threshold = 12 / state.transform.scale;
  for (let i = 0; i < state.elements.length; i++) {
    const elem = state.elements[i];
    if (elem.type === 'wall') {
      for (let j = 0; j < elem.points.length; j++) {
        const p = elem.points[j];
        const dx = pos.x - (p[0] * state.pixelsPerMeter);
        const dy = pos.y - (p[1] * state.pixelsPerMeter);
        if (Math.sqrt(dx * dx + dy * dy) < threshold) {
          return { elementIndex: i, pointIndex: j };
        }
      }
    }
  }
  return null;
}

function findWallAtPos(pos) {
  const threshold = 8 / state.transform.scale;
  for (let i = 0; i < state.elements.length; i++) {
    const elem = state.elements[i];
    if (elem.type === 'wall') {
      const p1 = { x: elem.points[0][0] * state.pixelsPerMeter, y: elem.points[0][1] * state.pixelsPerMeter };
      const p2 = { x: elem.points[1][0] * state.pixelsPerMeter, y: elem.points[1][1] * state.pixelsPerMeter };
      const dist = distToSegment(pos, p1, p2);
      if (dist < threshold) return i;
    }
  }
  return null;
}

function distToSegment(p, v, w) {
  const l2 = Math.pow(v.x - w.x, 2) + Math.pow(v.y - w.y, 2);
  if (l2 == 0) return Math.sqrt(Math.pow(p.x - v.x, 2) + Math.pow(p.y - v.y, 2));
  let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
  t = Math.max(0, Math.min(1, t));
  return Math.sqrt(Math.pow(p.x - (v.x + t * (w.x - v.x)), 2) + Math.pow(p.y - (v.y + t * (w.y - v.y)), 2));
}

function findApAtPos(pos) {
  const radius = 20; // Hit area radius
  for (let i = 0; i < state.accessPoints.length; i++) {
    const ap = state.accessPoints[i];
    const dx = pos.x - (ap.x * state.pixelsPerMeter);
    const dy = pos.y - (ap.y * state.pixelsPerMeter);
    if (Math.sqrt(dx * dx + dy * dy) < radius) {
      return i;
    }
  }
  return null;
}

// Initialize Select tool on load
window.addEventListener('load', () => {
  const selectBtn = document.getElementById('btnSelect');
  if (selectBtn) setActiveTool('select', selectBtn);
});
// Global Key Listeners
window.addEventListener('keydown', (e) => {
  // Don't delete if we are typing in an input or textarea
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (e.key === 'Delete' || e.key === 'Backspace') {
    deleteSelectedElement();
  }

  if (e.ctrlKey && e.key === 'z') {
    e.preventDefault();
    undo();
  }
  if (e.ctrlKey && e.key === 'y') {
    e.preventDefault();
    redo();
  }
});

function deleteSelectedElement() {
  if (state.selectedElementIndex !== null) {
    if (state.selectedElementType === 'wall') {
      state.elements.splice(state.selectedElementIndex, 1);
    } else if (state.selectedElementType === 'ap') {
      state.accessPoints.splice(state.selectedElementIndex, 1);
    }
    state.selectedElementIndex = null;
    state.selectedElementType = null;

    // Hide floating toolbar
    const toolbar = document.getElementById('floatingToolbar');
    if (toolbar) toolbar.style.display = 'none';

    invalidateHeatmap();
    saveHistory();
    draw();
  }
}

function updateFloatingToolbar() {
  const toolbar = document.getElementById('floatingToolbar');
  if (!toolbar) return;

  // Show toolbar if a wall or AP is SELECTED in select mode
  const isWall = state.selectedElementType === 'wall';
  const isAp = state.selectedElementType === 'ap';
  const idx = state.selectedElementIndex;

  if (idx !== null && (isWall || isAp) && state.mode === 'select' && !state.isDrawing && !state.draggingHandle) {
    let midX, midY;
    if (isWall) {
      const elem = state.elements[idx];
      const p1 = { x: elem.points[0][0] * state.pixelsPerMeter, y: elem.points[0][1] * state.pixelsPerMeter };
      const p2 = { x: elem.points[1][0] * state.pixelsPerMeter, y: elem.points[1][1] * state.pixelsPerMeter };
      midX = (p1.x + p2.x) / 2;
      midY = (p1.y + p2.y) / 2;
    } else {
      const ap = state.accessPoints[idx];
      midX = ap.x * state.pixelsPerMeter;
      midY = ap.y * state.pixelsPerMeter;
    }

    // Calculate screen position
    const screenX = midX * state.transform.scale + state.transform.x;
    const screenY = midY * state.transform.scale + state.transform.y - 50;

    toolbar.style.left = `${screenX}px`;
    toolbar.style.top = `${screenY}px`;
    toolbar.style.display = 'flex';
    toolbar.style.transform = 'translate(-50%, -50%)';

    // Toggle buttons based on type
    const btnReplace = document.getElementById('btnReplaceWall');
    if (btnReplace) btnReplace.style.display = isWall ? 'flex' : 'none';
  } else {
    toolbar.style.display = 'none';
  }
}

function copySelectedElement() {
  if (state.selectedElementIndex === null) return;

  if (state.selectedElementType === 'wall') {
    const original = state.elements[state.selectedElementIndex];
    const offset = 0.5; // 0.5 meters
    const copy = JSON.parse(JSON.stringify(original));
    copy.points = copy.points.map(p => [p[0] + offset, p[1] + offset]);
    state.elements.push(copy);
    state.selectedElementIndex = state.elements.length - 1;
  } else if (state.selectedElementType === 'ap') {
    const original = state.accessPoints[state.selectedElementIndex];
    const offset = 1.0;
    const copy = JSON.parse(JSON.stringify(original));
    copy.x += offset;
    copy.y += offset;
    copy.name += " (Copy)";
    state.accessPoints.push(copy);
    state.selectedElementIndex = state.accessPoints.length - 1;
  }

  invalidateHeatmap();
  saveHistory();
  draw();
}

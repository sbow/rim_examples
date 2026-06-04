#!/usr/bin/env node
// Process 3 — Node.js dashboard (RoboticsIpcModule peer `dashboard_feed`).
//
// Subscribes to the router over UDP, receives topic-20 "annotated frame ready"
// notifications, reads the annotated RGB frame + packed detections straight out
// of the SHM sideband rings, and streams them to a browser over WebSocket along
// with live pipeline + system-load stats. The router carries only 64 B control
// frames; the megabytes of pixels move peer-to-peer through /dev/shm.
//
// RIM surface used: the Node UDP bridge (RouterPeer / RouterFrame) reused
// verbatim from RoboticsIpcModule/examples/bridges/node_gateway, plus the
// shared frame-ring reader (../common/frame_shm.js).
//
// External deps (RIM stays dependency-free): `ws`, `smol-toml`.

import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { WebSocketServer } from 'ws';
import { parse as parseToml } from 'smol-toml';

import { FrameShmReader } from '../common/frame_shm.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const RIM_DIR = process.env.RIM_DIR ||
    path.resolve(__dirname, '../../../../../RoboticsIpcModule');
const { RouterPeer } = await import(
    path.join(RIM_DIR, 'examples/bridges/node_gateway/rim_router_peer.js'));

const PEER_ID = 8;
const TOPIC_ANNOTATED = 20;
const HTTP_PORT = Number(process.env.VCD_HTTP_PORT || 8080);
const DET_RECORD_BYTES = 24;

// --- parse the shared topology -----------------------------------------------
const cfgPath = process.argv[2] || path.resolve(__dirname, '../topology.toml');
const cfg = parseToml(fs.readFileSync(cfgPath, 'utf8'));

function splitUdp(addr) {
    const [, host, port] = addr.split(':');
    return { host, port: Number(port) };
}
const self = cfg.peers.find((p) => p.id === PEER_ID);
const me = splitUdp(self.local);
const router = splitUdp(cfg.router.listen_udp);
const ml = cfg.peers.find((p) => p.id === 5);
const annotatedName = ml.sideband[0].name;
const detectionsName = ml.sideband[1].name;
const width = cfg.video?.width ?? 640;
const height = cfg.video?.height ?? 360;

// --- HTTP + WebSocket server -------------------------------------------------
const server = http.createServer((req, res) => {
    const file = req.url === '/' ? 'index.html' : req.url.replace(/^\//, '');
    const full = path.join(__dirname, 'public', path.basename(file));
    fs.readFile(full, (err, data) => {
        if (err) { res.writeHead(404); res.end('not found'); return; }
        const type = full.endsWith('.html') ? 'text/html'
            : full.endsWith('.js') ? 'text/javascript' : 'text/plain';
        res.writeHead(200, { 'content-type': type });
        res.end(data);
    });
});
const wss = new WebSocketServer({ server });
server.listen(HTTP_PORT, () =>
    console.log(`[dashboard] http://localhost:${HTTP_PORT}  (router ${router.host}:${router.port})`));

function broadcast(data) {
    for (const c of wss.clients) if (c.readyState === 1) c.send(data);
}

// --- SHM readers -------------------------------------------------------------
const annotated = new FrameShmReader(annotatedName);
const detections = new FrameShmReader(detectionsName);

// --- rolling stats -----------------------------------------------------------
let frames = 0;
let lastFpsT = Date.now();
let fps = 0;
const stats = { fps: 0, infer_ms: 0, e2e_ms: 0, boxes: 0, backend: 'pending' };

// Frame binary header: [w u16][h u16][seq u32] then RGB24 pixels.
function packFrame(rgb, seq) {
    const head = Buffer.alloc(8);
    head.writeUInt16LE(width, 0);
    head.writeUInt16LE(height, 2);
    head.writeUInt32LE(seq >>> 0, 4);
    return Buffer.concat([head, rgb]);
}

function parseDetections(buf, n) {
    const out = [];
    for (let i = 0; i < n; i += 1) {
        const o = i * DET_RECORD_BYTES;
        out.push({
            x: buf.readFloatLE(o), y: buf.readFloatLE(o + 4),
            w: buf.readFloatLE(o + 8), h: buf.readFloatLE(o + 12),
            score: buf.readFloatLE(o + 16), cls: buf.readUInt32LE(o + 20),
        });
    }
    return out;
}

const peer = new RouterPeer({
    routerHost: router.host, routerPort: router.port,
    peerHost: me.host, peerPort: me.port, peerId: PEER_ID,
});

peer.on('frame', (frame) => {
    if (frame.topic_id !== TOPIC_ANNOTATED) return;

    const p = frame.payload; // 32 B inline AnnStats
    const nb = p.readUInt16LE(4);
    stats.infer_ms = p.readFloatLE(8);
    stats.e2e_ms = p.readFloatLE(16);
    stats.boxes = nb;

    const rgb = annotated.read(frame.sideband_seq);
    if (!rgb) return;

    let boxes = [];
    const det = detections.read(0n);
    if (det && nb > 0) boxes = parseDetections(det, Math.min(nb, 64));

    broadcast(packFrame(rgb, Number(frame.seq)));
    broadcast(JSON.stringify({ type: 'stats', ...stats, fps, boxes,
        load: os.loadavg(), cpu: cpuPercent() }));

    frames += 1;
    const dt = Date.now() - lastFpsT;
    if (dt >= 1000) { fps = (frames * 1000) / dt; stats.fps = fps; frames = 0; lastFpsT = Date.now(); }
});

// --- system CPU% (delta of /proc-derived os.cpus times) ----------------------
let prevCpu = os.cpus();
function cpuPercent() {
    const now = os.cpus();
    let idle = 0, total = 0;
    for (let i = 0; i < now.length; i += 1) {
        const a = prevCpu[i].times, b = now[i].times;
        const dIdle = b.idle - a.idle;
        const dTot = (b.user - a.user) + (b.nice - a.nice) + (b.sys - a.sys) +
            (b.irq - a.irq) + dIdle;
        idle += dIdle; total += dTot;
    }
    prevCpu = now;
    return total > 0 ? (1 - idle / total) * 100 : 0;
}

await peer.open();
console.log(`[dashboard] peer ${me.host}:${me.port} id=${PEER_ID} bound; ` +
    `reading ${annotatedName} + ${detectionsName}`);

for (const sig of ['SIGINT', 'SIGTERM']) {
    process.on(sig, async () => {
        console.log('[dashboard] shutting down');
        annotated.close(); detections.close();
        await peer.close();
        process.exit(0);
    });
}

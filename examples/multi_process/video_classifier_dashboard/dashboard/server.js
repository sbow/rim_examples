/**
 * Process 3: Dashboard Server
 *
 * Node.js web server that reads annotated frames and metadata from shared memory,
 * streams them to connected browsers via WebSocket, and displays a visually
 * rich dashboard with processed video and performance statistics.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');

// Configuration
const PORT = parseInt(process.env.DASHBOARD_PORT || '3000', 10);
const SHM_ANNOTATED = '/dev/shm/vcf_annotated_frames';
const SHM_METADATA = '/dev/shm/vcf_metadata';
const SHM_HEADER_SIZE = 64;
const FRAME_POLL_INTERVAL_MS = 33; // ~30 FPS

/**
 * Shared memory reader for Node.js.
 * Reads frames and metadata published by the YOLO classifier.
 */
class ShmReader {
    constructor(shmPath) {
        this.shmPath = shmPath;
        this.fd = null;
        this.buffer = null;
        this.lastSequence = BigInt(0);
    }

    connect() {
        try {
            this.fd = fs.openSync(this.shmPath, 'r');
            console.log(`[Dashboard] Connected to SHM: ${this.shmPath}`);
            return true;
        } catch (err) {
            return false;
        }
    }

    readHeader() {
        if (this.fd === null) return null;

        const headerBuf = Buffer.alloc(SHM_HEADER_SIZE);
        try {
            fs.readSync(this.fd, headerBuf, 0, SHM_HEADER_SIZE, 0);
        } catch (err) {
            return null;
        }

        const sequence = headerBuf.readBigUInt64LE(0);
        const timestampNs = headerBuf.readBigUInt64LE(8);
        const width = headerBuf.readUInt32LE(16);
        const height = headerBuf.readUInt32LE(20);
        const channels = headerBuf.readUInt32LE(24);
        const dataSize = headerBuf.readUInt32LE(28);

        return { sequence, timestampNs, width, height, channels, dataSize };
    }

    readFrame() {
        const header = this.readHeader();
        if (!header || header.sequence <= this.lastSequence) return null;
        if (header.width === 0 || header.dataSize === 0) return null;

        const dataBuf = Buffer.alloc(header.dataSize);
        try {
            fs.readSync(this.fd, dataBuf, 0, header.dataSize, SHM_HEADER_SIZE);
        } catch (err) {
            return null;
        }

        this.lastSequence = header.sequence;
        return { ...header, data: dataBuf };
    }

    readMetadata() {
        const header = this.readHeader();
        if (!header || header.sequence <= this.lastSequence) return null;
        if (header.dataSize === 0) return null;

        const dataBuf = Buffer.alloc(header.dataSize);
        try {
            fs.readSync(this.fd, dataBuf, 0, header.dataSize, SHM_HEADER_SIZE);
        } catch (err) {
            return null;
        }

        this.lastSequence = header.sequence;
        try {
            return JSON.parse(dataBuf.toString('utf-8'));
        } catch {
            return null;
        }
    }

    close() {
        if (this.fd !== null) {
            fs.closeSync(this.fd);
            this.fd = null;
        }
    }
}

/**
 * Performance statistics tracker.
 */
class StatsTracker {
    constructor(windowSize = 100) {
        this.windowSize = windowSize;
        this.inferenceTimes = [];
        this.totalTimes = [];
        this.frameTimestamps = [];
        this.detectionCounts = [];
        this.classCounts = {};
    }

    update(metadata) {
        const now = Date.now();
        this.frameTimestamps.push(now);
        this.inferenceTimes.push(metadata.inference_ms);
        this.totalTimes.push(metadata.total_ms);
        this.detectionCounts.push(metadata.num_detections);

        // Track class frequencies
        if (metadata.detections) {
            for (const det of metadata.detections) {
                this.classCounts[det.class] = (this.classCounts[det.class] || 0) + 1;
            }
        }

        // Trim to window size
        if (this.inferenceTimes.length > this.windowSize) {
            this.inferenceTimes.shift();
            this.totalTimes.shift();
            this.frameTimestamps.shift();
            this.detectionCounts.shift();
        }
    }

    getStats() {
        if (this.inferenceTimes.length === 0) {
            return {
                fps: 0,
                avgInferenceMs: 0,
                avgTotalMs: 0,
                avgDetections: 0,
                topClasses: [],
                history: { inference: [], total: [], detections: [] }
            };
        }

        const len = this.frameTimestamps.length;
        const timeSpan = (this.frameTimestamps[len - 1] - this.frameTimestamps[0]) / 1000;
        const fps = timeSpan > 0 ? (len - 1) / timeSpan : 0;

        const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

        // Top detected classes
        const topClasses = Object.entries(this.classCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .map(([name, count]) => ({ name, count }));

        return {
            fps: Math.round(fps * 10) / 10,
            avgInferenceMs: Math.round(avg(this.inferenceTimes) * 100) / 100,
            avgTotalMs: Math.round(avg(this.totalTimes) * 100) / 100,
            avgDetections: Math.round(avg(this.detectionCounts) * 10) / 10,
            topClasses,
            history: {
                inference: this.inferenceTimes.slice(-50),
                total: this.totalTimes.slice(-50),
                detections: this.detectionCounts.slice(-50)
            }
        };
    }
}

// Serve static HTML dashboard
const HTML_CONTENT = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf-8');

const server = http.createServer((req, res) => {
    if (req.url === '/' || req.url === '/index.html') {
        res.writeHead(200, { 'Content-Type': 'text/html' });
        res.end(HTML_CONTENT);
    } else {
        res.writeHead(404);
        res.end('Not Found');
    }
});

// WebSocket server for real-time streaming
const wss = new WebSocketServer({ server });
const stats = new StatsTracker();

// SHM readers
const frameReader = new ShmReader(SHM_ANNOTATED);
const metaReader = new ShmReader(SHM_METADATA);

// Retry connection to SHM
function connectReaders() {
    let connected = false;
    const interval = setInterval(() => {
        if (frameReader.connect() && metaReader.connect()) {
            connected = true;
            clearInterval(interval);
            console.log('[Dashboard] Connected to shared memory channels');
            startStreaming();
        } else {
            console.log('[Dashboard] Waiting for shared memory...');
        }
    }, 1000);
}

// Frame streaming loop
function startStreaming() {
    setInterval(() => {
        // Read metadata
        const metadata = metaReader.readMetadata();
        if (metadata) {
            stats.update(metadata);
        }

        // Read annotated frame
        const frame = frameReader.readFrame();
        if (!frame) return;

        // Encode as JPEG-like raw format for WebSocket transmission
        // In production, use a proper encoder; here we send raw RGB + metadata
        const statsData = stats.getStats();
        const message = JSON.stringify({
            type: 'frame',
            width: frame.width,
            height: frame.height,
            channels: frame.channels,
            stats: statsData,
            metadata: metadata
        });

        // Broadcast to all connected clients
        for (const client of wss.clients) {
            if (client.readyState === 1) { // WebSocket.OPEN
                // Send metadata as text
                client.send(message);
                // Send frame data as binary
                client.send(frame.data);
            }
        }
    }, FRAME_POLL_INTERVAL_MS);
}

wss.on('connection', (ws) => {
    console.log('[Dashboard] Client connected');
    ws.on('close', () => console.log('[Dashboard] Client disconnected'));
});

// Start server
server.listen(PORT, () => {
    console.log(`=== Video Classifier Dashboard (Process 3) ===`);
    console.log(`[Dashboard] Server running at http://localhost:${PORT}`);
    connectReaders();
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n[Dashboard] Shutting down...');
    frameReader.close();
    metaReader.close();
    wss.close();
    server.close();
    process.exit(0);
});

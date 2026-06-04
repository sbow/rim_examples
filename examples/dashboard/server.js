const express = require('express');
const http = require('http');
const fs = require('fs');
const unixDgram = require('unix-dgram');
const { WebSocketServer } = require('ws');

const udsPath = process.env.RIM_DASHBOARD_UDS || '/tmp/rim_examples_dashboard.sock';
const port = Number(process.env.PORT || 8080);

if (fs.existsSync(udsPath)) {
  fs.unlinkSync(udsPath);
}

const app = express();
app.use(express.static('public'));

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

function decodeFrame(buffer) {
  return {
    source: buffer.readUInt8(0),
    topic: buffer.readUInt16LE(2),
    seq: buffer.readUInt32LE(4),
    timestampNs: Number(buffer.readBigUInt64LE(8)),
    payload: buffer.subarray(32, 64).toString('utf8').replace(/\0+$/, ''),
  };
}

const uds = unixDgram.createSocket('unix_dgram');
uds.on('message', (msg) => {
  if (msg.length < 64) {
    return;
  }

  const frame = decodeFrame(msg.subarray(0, 64));
  const payload = JSON.stringify(frame);

  for (const client of wss.clients) {
    if (client.readyState === 1) {
      client.send(payload);
    }
  }
});

uds.bind(udsPath);

server.listen(port, () => {
  console.log(`RIM dashboard listening on http://localhost:${port}`);
  console.log(`Attach a router peer to UDS ${udsPath}`);
});

function shutdown() {
  uds.close();
  server.close(() => {
    if (fs.existsSync(udsPath)) {
      fs.unlinkSync(udsPath);
    }
    process.exit(0);
  });
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

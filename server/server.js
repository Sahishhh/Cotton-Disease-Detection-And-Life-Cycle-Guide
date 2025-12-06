const express = require('express');
const multer = require('multer');
const path = require('path');
const { spawnSync } = require('child_process');
const fs = require('fs');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadDir),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const name = Date.now() + '-' + Math.round(Math.random() * 1E9) + ext;
    cb(null, name);
  }
});
const upload = multer({ storage });

// Health check endpoint
app.get('/api/health', (req, res) => res.json({ status: 'ok' }));

// Prediction endpoint
app.post('/api/predict', upload.single('image'), (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No file uploaded' });
    const imagePath = req.file.path;

    // Python script location (inside model folder)
    const pyScript = path.join(__dirname, '..', 'model', 'predict.py');
    const modelDir = path.join(__dirname, '..', 'model');

    // Run Python script synchronously
    const python = spawnSync('python', [pyScript, imagePath], {
      encoding: 'utf8',
      cwd: modelDir,
      maxBuffer: 20 * 1024 * 1024
    });

    if (python.error) {
      console.error('Python spawn error:', python.error);
      return res.status(500).json({ error: python.error.message });
    }

    const stdout = python.stdout ? python.stdout.trim() : '';
    const stderr = python.stderr ? python.stderr.trim() : '';

    if (stderr) console.error('python stderr:', stderr);

    let result;
    try {
      // Extract only the JSON part from TensorFlow output
      const jsonMatch = stdout.match(/\{.*\}/s);
      if (!jsonMatch) throw new Error('No JSON found in Python output');
      result = JSON.parse(jsonMatch[0]);
    } catch (err) {
      console.error('Failed to parse python output', err, 'stdout:', stdout);
      return res.status(500).json({ error: 'Invalid model output', raw: stdout, stderr });
    }

    // Optionally delete the uploaded file
    // fs.unlinkSync(imagePath);

    return res.json({ prediction: result });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`✅ Server running on http://localhost:${PORT}`));

import React, { useState } from 'react';
import { Github, UploadCloud, Loader2, AlertCircle, Image, Video, Mic, X, CheckCircle, XCircle } from 'lucide-react';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const PHOTO_THRESHOLD = 0.25
const AUDIO_THRESHOLD = 0.5;

// Header Component
const Header = () => {
  return (
    <nav className="bg-gradient-to-r from-gray-900 via-gray-800 to-gray-900 shadow-2xl border-b border-gray-700">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex justify-between items-center h-20">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-indigo-400 to-purple-400 bg-clip-text text-transparent">
              Multi-Modal Deepfake Detector
            </h1>
            <p className="text-xs text-gray-400 mt-1">AI-Powered Media Authentication</p>
          </div>
        </div>
      </div>
    </nav>
  );
};

// Mode Selector Component
const modes = [
  { id: 'photo', name: 'Photo', Icon: Image },
  { id: 'video', name: 'Video', Icon: Video },
  { id: 'audio', name: 'Audio', Icon: Mic },
];

const ModeSelector = ({ mode, setMode, disabled }) => {
  return (
    <div className="flex justify-center space-x-3 p-3 bg-gray-800/50 rounded-xl border border-gray-700">
      {modes.map((m) => {
        const isActive = mode === m.id;
        const Icon = m.Icon;
        return (
          <button
            key={m.id}
            onClick={() => !disabled && setMode(m.id)}
            disabled={disabled}
            className={`
              flex items-center space-x-2 px-8 py-3 rounded-lg font-medium text-sm
              transition-all duration-200 ease-in-out transform
              ${isActive
                ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg scale-105'
                : 'text-gray-400 hover:bg-gray-700 hover:text-white hover:scale-102'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            <Icon size={20} />
            <span>{m.name}</span>
          </button>
        );
      })}
    </div>
  );
};

// File Preview Component
const FilePreview = ({ file, mode, onRemove }) => {
  const [previewUrl, setPreviewUrl] = useState(null);

  React.useEffect(() => {
    if (file && (mode === 'photo' || mode === 'video')) {
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [file, mode]);

  if (!file) return null;

  return (
    <div className="mt-6 p-6 bg-gray-800/50 rounded-xl border border-gray-700">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-gray-200">File Preview</h3>
        <button
          onClick={onRemove}
          className="p-2 hover:bg-gray-700 rounded-lg transition-colors"
          title="Remove file"
        >
          <X size={20} className="text-gray-400" />
        </button>
      </div>
      
      <div className="space-y-3">
        {mode === 'photo' && previewUrl && (
          <div className="relative rounded-lg overflow-hidden bg-gray-900">
            <img 
              src={previewUrl} 
              alt="Preview" 
              className="w-full h-64 object-contain"
            />
          </div>
        )}
        
        {mode === 'video' && previewUrl && (
          <div className="relative rounded-lg overflow-hidden bg-gray-900">
            <video 
              src={previewUrl} 
              controls 
              className="w-full h-64 object-contain"
            />
          </div>
        )}
        
        {mode === 'audio' && (
          <div className="flex items-center justify-center h-32 bg-gray-900 rounded-lg">
            <div className="text-center">
              <Mic size={48} className="mx-auto text-indigo-400 mb-2" />
              <p className="text-gray-300 font-medium">{file.name}</p>
              <p className="text-xs text-gray-500 mt-1">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
        )}
        
        <div className="flex items-center justify-between text-sm bg-gray-900 p-3 rounded-lg">
          <span className="text-gray-400">Filename:</span>
          <span className="text-gray-200 font-mono text-xs truncate max-w-xs ml-2">
            {file.name}
          </span>
        </div>
      </div>
    </div>
  );
};

// Reusable Confidence Bar Component
const ConfidenceBar = ({ title, probability, description, isMainResult = false, mainIsFake = false }) => {
  
  let probValue, percentage, isFake, colorClass;

  if (probability === undefined || probability === null || probability === "N/A") {
    // Handle N/A audio
    return (
      <div className="pt-4 border-t border-gray-700">
        <div className="flex justify-between text-sm text-gray-300 mb-2">
          <span className="font-medium">{title}</span>
          <span className="font-mono font-bold text-gray-500">N/A</span>
        </div>
        <div className="relative w-full bg-gray-700/50 rounded-full h-4 overflow-hidden">
          <div className="h-full rounded-full bg-gray-600" style={{ width: '100%' }}></div>
        </div>
        {description && <p className="text-xs text-gray-500 mt-2">{description}</p>}
      </div>
    );
  }

  probValue = probability;
  percentage = probValue * 100;
  
  if (isMainResult) {
    // The main bar's color is based on the final label
    isFake = mainIsFake;
  } else {
    // Sub-bars (visual/audio) have their color based on their own value
    const threshold = title.toLowerCase().includes('audio') ? AUDIO_THRESHOLD : PHOTO_THRESHOLD;
    isFake = probValue >= threshold;
  }

  colorClass = isFake
    ? 'bg-gradient-to-r from-red-600 to-red-500'
    : 'bg-gradient-to-r from-green-600 to-green-500';

  return (
    <div className={isMainResult ? "" : "pt-4 border-t border-gray-700"}>
      <div className="flex justify-between text-sm text-gray-300 mb-2">
        <span className="font-medium">{title}</span>
        <span className="font-mono font-bold">{percentage.toFixed(2)}%</span>
      </div>
      <div className="relative w-full bg-gray-700/50 rounded-full h-4 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out ${colorClass}`}
          style={{ width: `${percentage}%` }}
        >
          <div className="h-full w-full opacity-50 bg-gradient-to-r from-transparent via-white to-transparent animate-shimmer"></div>
        </div>
      </div>
      {description && <p className="text-xs text-gray-500 mt-2">{description}</p>}
    </div>
  );
};

// Prediction Result Component
const PredictionResult = ({ result }) => {
  if (!result) return null;

  const { 
    label, 
    probability_fake, // For photo/audio
    probability_fake_avg, // For video (final)
    frames_with_face,
    visual_prob, // For video (sub)
    audio_prob  // For video (sub)
  } = result;
  
  const isVideo = probability_fake_avg !== undefined;
  const isFake = label === 'Fake';
  const Icon = isFake ? XCircle : CheckCircle;
  
  // Determine the final probability to display
  const finalProbValue = isVideo ? probability_fake_avg : probability_fake;
  
  return (
    <div className="mt-8 animate-fadeIn">
      <div className={`p-8 rounded-xl border-2 ${
        isFake 
          ? 'bg-red-500/10 border-red-500/50' 
          : 'bg-green-500/10 border-green-500/50'
      } shadow-2xl`}>
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-xl font-bold text-white">Analysis Result</h3>
          <div className={`flex items-center space-x-2 px-5 py-2 rounded-full ${
            isFake ? 'bg-red-500/20' : 'bg-green-500/20'
          }`}>
            <Icon size={24} className={isFake ? 'text-red-400' : 'text-green-400'} />
            <span className={`text-lg font-bold ${
              isFake ? 'text-red-400' : 'text-green-400'
            }`}>
              {label}
            </span>
          </div>
        </div>

        <div className="space-y-4">
          
          {/* --- Main Confidence Score --- */}
          <ConfidenceBar
            title={isVideo ? "Final Confidence Score (Weighted)" : "Confidence Score"}
            probability={finalProbValue}
            isMainResult={true}
            mainIsFake={isFake}
            description={
              isFake 
                ? 'High probability of manipulation detected' 
                : 'Media appears to be authentic'
            }
          />

          {/* --- Video-Specific Sub-Scores --- */}
          {isVideo && (
            <>
              <ConfidenceBar
                title="Visual Confidence"
                probability={visual_prob}
              />
              <ConfidenceBar
                title="Audio Confidence"
                probability={audio_prob}
                description={audio_prob === "N/A" ? "Audio track not found or silent." : null}
              />
            </>
          )}

          {/* --- Frames Analyzed (for video) --- */}
          {frames_with_face !== undefined && (
            <div className="pt-4 border-t border-gray-700">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-400">Frames Analyzed</span>
                <span className="text-indigo-400 font-bold">{frames_with_face}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};


// Main App Component
function App() {
  const [mode, setMode] = useState('photo');
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [frameRate, setFrameRate] = useState(1.0);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (selectedFile) => {
    if (selectedFile) {
      setFile(selectedFile);
      setResult(null);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileChange(droppedFile);
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setFile(null);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
      e.preventDefault();
      if (!file) {
      setError('Please choose a file first.');
      return;
    }

    setIsLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    let endpoint = '';
    let url = '';

    if (mode === 'photo') {
      endpoint = '/predict/photo';
      url = `${API_URL}${endpoint}`;
    } else if (mode === 'video') {
      endpoint = '/predict/video';
      url = `${API_URL}${endpoint}?frame_rate=${frameRate}`;
    } else if (mode === 'audio') {
      endpoint = '/predict/audio';
      url = `${API_URL}${endpoint}`;
    }

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.detail || 'An unknown error occurred.');
      }
      setResult(data);

    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const fileAccept = {
    photo: 'image/*',
    video: 'video/*',
    audio: 'audio/*',
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-gray-100">
      <Header />
      
      <main className="max-w-4xl mx-auto p-6 pb-32">
        <div className="bg-gray-800/30 backdrop-blur-sm shadow-2xl rounded-2xl p-8 border border-gray-700">
          <ModeSelector mode={mode} setMode={handleModeChange} disabled={isLoading} />

          <div className="mt-8">
            {!file ? (
              <label
                htmlFor="file-upload"
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative flex flex-col items-center justify-center w-full h-56 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 ${
                  isDragging
                    ? 'border-indigo-500 bg-indigo-500/10 scale-102'
                    : 'border-gray-600 bg-gray-800/50 hover:bg-gray-700/50 hover:border-gray-500'
                }`}
              >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <UploadCloud className={`w-16 h-16 mb-4 transition-colors ${
                    isDragging ? 'text-indigo-400' : 'text-gray-400'
                  }`} />
                  <p className="mb-2 text-base text-gray-300">
                    <span className="font-semibold text-indigo-400">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-sm text-gray-500">
                    {mode === 'photo' && 'PNG, JPG, JPEG (Max 10MB)'}
                    {mode === 'video' && 'MP4, MOV, AVI (Max 50MB)'}
                    {mode === 'audio' && 'WAV, MP3 (Max 10MB)'}
                  </p>
                </div>
                <input
                  id="file-upload"
                  type="file"
                  className="hidden"
                  accept={fileAccept[mode]}
                  onChange={(e) => handleFileChange(e.target.files[0])}
                  disabled={isLoading}
                />
              </label>
            ) : (
              <FilePreview file={file} mode={mode} onRemove={handleRemoveFile} />
            )}

            {mode === 'video' && file && (
              <div className="mt-6 p-6 bg-gray-800/50 rounded-xl border border-gray-700">
                <label htmlFor="frameRate" className="block mb-3 text-sm font-medium text-gray-300">
                  Frame Sampling Rate: <span className="text-indigo-400 font-bold">{frameRate.toFixed(1)} fps</span>
                </label>
                <input
                  id="frameRate"
                  type="range"
                  min="0.1"
                  max="5.0"
                  step="0.1"
                  value={frameRate}
                  onChange={(e) => setFrameRate(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                  disabled={isLoading}
                />
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                  <span>0.1 fps</span>
                  <span>5.0 fps</span>
                </div>
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={isLoading || !file}
              className="w-full mt-8 flex items-center justify-center px-6 py-4 text-lg font-semibold text-white bg-gradient-to-r from-indigo-600 to-purple-600 rounded-xl shadow-lg
                         hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 focus:ring-offset-gray-900
                         disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-102 active:scale-98"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-3 h-6 w-6 animate-spin" />
                  Analyzing {mode}...
                </>
              ) : (
                `Analyze ${mode.charAt(0).toUpperCase() + mode.slice(1)}`
              )}
            </button>
          </div>

          {error && (
            <div className="mt-6 flex items-start p-4 text-sm text-red-400 bg-red-500/20 rounded-xl border border-red-500/50 animate-fadeIn">
              <AlertCircle className="flex-shrink-0 mr-3 w-5 h-5 mt-0.5" />
              <div>
                <span className="font-semibold">Error:</span> {error}
              </div>
            </div>
          )}
          <PredictionResult result={result} />
        </div>
      </main>

      <footer className="fixed bottom-0 left-0 right-0 bg-gray-900/95 backdrop-blur-sm border-t border-gray-800 py-4">
        <div className="max-w-4xl mx-auto px-6 flex justify-between items-center">
          <p className="text-sm text-gray-400">
            Powered by AI Deep Learning Models
          </p>
          <a
            href="https://github.com/AryakumarMishra/multimodal-deepfake-detector"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-2 text-gray-400 hover:text-white transition-colors"
          >
            <Github size={20} />
            <span className="text-sm font-medium">View on GitHub</span>
          </a>
        </div>
      </footer>

      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-fadeIn {
          animation: fadeIn 0.5s ease-out;
        }
        .animate-shimmer {
          animation: shimmer 2s infinite;
        }
        .scale-102 {
          transform: scale(1.02);
        }
        .active\\:scale-98:active {
          transform: scale(0.98);
        }
      `}</style>
    </div>
  );
}

export default App;
import React from 'react';

const PredictionResult = ({ result }) => {
  if (!result) return null;

  const { label, probability_fake, probability_fake_avg, frames_with_face } = result;
  
  // Use the correct probability (video has a different key)
  const prob = probability_fake !== undefined ? probability_fake : probability_fake_avg;
  const isFake = label === 'Fake';
  const color = isFake ? 'red' : 'green';
  const percentage = (prob * 100).toFixed(2);

  return (
    <div className={`mt-6 p-6 border-l-4 border-${color}-500 bg-gray-800 rounded-r-lg shadow-xl`}>
      <div className="flex justify-between items-center">
        <span className="text-lg font-medium text-white">Prediction Result:</span>
        <span 
          className={`px-4 py-1 text-sm font-bold rounded-full ${isFake ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}
        >
          {label}
        </span>
      </div>

      <div className="mt-4">
        <div className="flex justify-between text-sm text-gray-300 mb-1">
          <span>{isFake ? 'Fake' : 'Real'} Probability</span>
          <span>{percentage}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2.5">
          <div
            className={`bg-${color}-500 h-2.5 rounded-full transition-all duration-500`}
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
      </div>

      {frames_with_face && (
        <p className="text-sm text-gray-400 mt-4">
          Analyzed {frames_with_face} frames containing a face.
        </p>
      )}
    </div>
  );
};

export default PredictionResult;
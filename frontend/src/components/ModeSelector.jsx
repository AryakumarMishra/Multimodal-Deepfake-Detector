import React from 'react';
import { Image, Video, Mic } from 'lucide-react';

const modes = [
  { id: 'photo', name: 'Photo', icon: <Image size={20} /> },
  { id: 'video', name: 'Video', icon: <Video size={20} /> },
  { id: 'audio', name: 'Audio', icon: <Mic size={20} /> },
];

const ModeSelector = ({ mode, setMode }) => {
  return (
    <div className="flex justify-center space-x-2 p-2 bg-gray-800 rounded-lg">
      {modes.map((m) => {
        const isActive = mode === m.id;
        return (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`
              flex items-center space-x-2 px-6 py-3 rounded-md font-medium text-sm
              transition-all duration-200 ease-in-out
              ${isActive
                ? 'bg-indigo-600 text-white shadow-lg'
                : 'text-gray-400 hover:bg-gray-700 hover:text-white'
              }
            `}
          >
            {m.icon}
            <span>{m.name}</span>
          </button>
        );
      })}
    </div>
  );
};

export default ModeSelector;
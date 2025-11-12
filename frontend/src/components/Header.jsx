import React from 'react';
import { Github } from 'lucide-react';

const Header = () => {
  return (
    <nav className="bg-gray-900 shadow-lg">
      <div className="max-w-6xl mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <span className="text-2xl font-semibold text-white">
            Multi-Modal Deepfake Detector
          </span>
          <a
            href="https://github.com/your-repo-link" // <-- Change this to your repo
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-400 hover:text-white"
          >
            <Github size={28} />
          </a>
        </div>
      </div>
    </nav>
  );
};

export default Header;
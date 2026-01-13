// frontend/src/components/Navbar.jsx
import React, { useEffect } from 'react';
import { motion } from 'framer-motion';

export default function Navbar({ onToggleSidebar, darkMode, setDarkMode }) {
    useEffect(() => {
        document.documentElement.classList.toggle('dark', darkMode);
        localStorage.setItem('darkMode', darkMode ? '1' : '0');
    }, [darkMode]);

    return (
        <nav className="flex items-center justify-between mb-6 p-4">
            <div>
                <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-pink-500 to-indigo-600">
                    Employee Attrition Predictor
                </h1>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    ML-Powered Workforce Analytics
                </p>
            </div>
            <div className="flex items-center gap-4">
                <button
                    onClick={() => onToggleSidebar()}
                    className="px-4 py-2 rounded-lg bg-indigo-100 hover:bg-indigo-200 dark:bg-indigo-900 dark:hover:bg-indigo-800 dark:text-indigo-100 font-semibold transition-all"
                >
                    View Details
                </button>

                {/* Dark Mode Toggle Switch */}
                <div className="flex items-center gap-3">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                        {darkMode ? 'Dark' : 'Light'}
                    </span>
                    <button
                        onClick={() => setDarkMode(!darkMode)}
                        className={`relative w-14 h-7 rounded-full transition-colors duration-300 ${
                            darkMode ? 'bg-indigo-600' : 'bg-gray-300'
                        }`}
                    >
                        <motion.div
                            className="absolute top-1 left-1 w-5 h-5 bg-white rounded-full shadow-md"
                            animate={{ x: darkMode ? 28 : 0 }}
                            transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                        />
                    </button>
                </div>
            </div>
        </nav>
    );
}

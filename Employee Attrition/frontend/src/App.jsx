import React from 'react';
import Dashboard from './components/Dashboard';

export default function App() {
    return (
        <div className="min-h-screen p-6 dark:text-gray-100">
            <header className="mb-6">
                <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-pink-500 to-indigo-600">Employee Attrition Predictor</h1>
                <p className="text-sm text-gray-600 dark:text-gray-300">Interactive dashboard powered by ML — explore metrics and make live predictions</p>
            </header>
            <main>
                <Dashboard />
            </main>
        </div>
    )
}

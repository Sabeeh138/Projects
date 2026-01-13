// frontend/src/components/Dashboard.jsx
import React, { useEffect, useState } from 'react';
import { getMetrics, predict, recommend, limeExplain } from '../services/api';
import PredictForm from './PredictForm';
import Charts from './Charts';
import Navbar from './Navbar';
import FeatureImportance from './FeatureImportance';
import ShapSummary from './ShapSummary';
import ModelRadarChart from './ModelRadarChart';
import { motion, AnimatePresence } from 'framer-motion';

export default function Dashboard() {
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [lastPrediction, setLastPrediction] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [darkMode, setDarkMode] = useState(() => localStorage.getItem('darkMode') === '1');
    const [predicting, setPredicting] = useState(false);
    const [recs, setRecs] = useState(null);
    const [lime, setLime] = useState(null);

    useEffect(() => {
        async function load() {
            try {
                const data = await getMetrics();
                setMetrics(data);
            } catch (err) {
                console.error(err);
            } finally { setLoading(false); }
        }
        load();
    }, []);

    const handlePredict = async (payload) => {
        setPredicting(true);
        try {
            const res = await predict(payload);
            let recData = null;
            let limeData = null;
            try {
                recData = await recommend(payload);
            } catch (e) { console.error('Recommend error', e); }
            try {
                limeData = await limeExplain(payload);
            } catch (e) { console.error('LIME error', e); }
            // Add small delay for animation effect
            setTimeout(() => {
                setLastPrediction(res);
                setRecs(recData);
                setLime(limeData);
                setPredicting(false);
            }, 800);
        } catch (err) {
            console.error(err);
            alert("Prediction error: " + (err?.response?.data?.error || err.message));
            setPredicting(false);
        }
    };

    // Calculate best model
    const bestModel = metrics ? Object.entries(metrics).reduce((best, [name, data]) =>
        data.roc_auc > (best.auc || 0) ? { name, auc: data.roc_auc } : best
        , {}) : null;

    return (
        <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
            <Navbar onToggleSidebar={() => setSidebarOpen(v => !v)} darkMode={darkMode} setDarkMode={setDarkMode} />

            <div className="container mx-auto px-4 py-8">
                {/* Stats Cards */}
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8"
                >
                    <motion.div
                        whileHover={{ scale: 1.05, y: -5 }}
                        className="bg-gradient-to-br from-indigo-500 to-purple-600 p-6 rounded-2xl shadow-xl text-white relative overflow-hidden"
                    >
                        <div className="absolute top-0 right-0 w-32 h-32 bg-white opacity-10 rounded-full -mr-16 -mt-16"></div>
                        <div className="relative z-10">
                            <p className="text-sm opacity-90 mb-1">Best Model</p>
                            <h3 className="text-3xl font-bold">{bestModel?.name || 'Loading...'}</h3>
                            <p className="text-sm mt-2 opacity-90">AUC: {(bestModel?.auc * 100).toFixed(2)}%</p>
                        </div>
                    </motion.div>

                    <motion.div
                        whileHover={{ scale: 1.05, y: -5 }}
                        className="bg-gradient-to-br from-pink-500 to-rose-600 p-6 rounded-2xl shadow-xl text-white relative overflow-hidden"
                    >
                        <div className="absolute top-0 right-0 w-32 h-32 bg-white opacity-10 rounded-full -mr-16 -mt-16"></div>
                        <div className="relative z-10">
                            <p className="text-sm opacity-90 mb-1">Models Trained</p>
                            <h3 className="text-3xl font-bold">{metrics ? Object.keys(metrics).length : 0}</h3>
                            <p className="text-sm mt-2 opacity-90">Ensemble Methods</p>
                        </div>
                    </motion.div>

                    <motion.div
                        whileHover={{ scale: 1.05, y: -5 }}
                        className="bg-gradient-to-br from-emerald-500 to-teal-600 p-6 rounded-2xl shadow-xl text-white relative overflow-hidden"
                    >
                        <div className="absolute top-0 right-0 w-32 h-32 bg-white opacity-10 rounded-full -mr-16 -mt-16"></div>
                        <div className="relative z-10">
                            <p className="text-sm opacity-90 mb-1">Predictions Made</p>
                            <h3 className="text-3xl font-bold">{lastPrediction ? '1+' : '0'}</h3>
                            <p className="text-sm mt-2 opacity-90">Real-time Analysis</p>
                        </div>
                    </motion.div>
                </motion.div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2 space-y-6">
                        {/* Model Metrics */}
                        <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            whileHover={{ scale: 1.01 }}
                            className="bg-white/80 backdrop-blur-lg p-6 rounded-2xl shadow-xl border border-white/20 dark:bg-gray-800/80 dark:border-gray-700/20"
                        >
                            <h2 className="text-2xl font-bold mb-4 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                                Model Performance
                            </h2>
                            {loading ? (
                                <div className="flex items-center justify-center h-64">
                                    <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-indigo-600"></div>
                                </div>
                            ) : metrics ? <Charts metrics={metrics} /> : <p>No metrics available</p>}
                        </motion.div>

                        {/* Feature Importance & Radar */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.1 }}
                                whileHover={{ scale: 1.02 }}
                            >
                                <FeatureImportance />
                            </motion.div>
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.2 }}
                                whileHover={{ scale: 1.02 }}
                            >
                                <ModelRadarChart metrics={metrics || {}} />
                            </motion.div>
                        </div>
                    </div>

                    <div className="lg:col-span-1 space-y-6">
                        {/* Prediction Form */}
                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="bg-white/80 backdrop-blur-lg p-6 rounded-2xl shadow-xl border border-white/20 dark:bg-gray-800/80 dark:border-gray-700/20"
                        >
                            <h2 className="text-2xl font-bold mb-4 bg-gradient-to-r from-pink-600 to-rose-600 bg-clip-text text-transparent">
                                Make a Prediction
                            </h2>
                            <PredictForm onPredict={handlePredict} predicting={predicting} />

                            <AnimatePresence mode="wait">
                                {predicting && (
                                    <motion.div
                                        initial={{ opacity: 0, scale: 0.8 }}
                                        animate={{ opacity: 1, scale: 1 }}
                                        exit={{ opacity: 0, scale: 0.8 }}
                                        className="mt-4 p-4 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl text-white text-center"
                                    >
                                        <div className="animate-pulse">Analyzing...</div>
                                    </motion.div>
                                )}

                                {lastPrediction && !predicting && (
                                    <motion.div
                                        initial={{ opacity: 0, scale: 0.8, y: 20 }}
                                        animate={{ opacity: 1, scale: 1, y: 0 }}
                                        exit={{ opacity: 0, scale: 0.8, y: -20 }}
                                        className="mt-4 p-6 bg-gradient-to-br from-indigo-50 to-pink-50 border-2 border-indigo-200 rounded-2xl shadow-lg dark:from-gray-700 dark:to-gray-800 dark:border-gray-600"
                                    >
                                        {/* Probability Gauge */}
                                        <div className="mb-4">
                                            <div className="flex justify-between items-center mb-2">
                                                <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">Attrition Risk</span>
                                                <span className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                                                    {(lastPrediction.probability * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                            <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden dark:bg-gray-700">
                                                <motion.div
                                                    initial={{ width: 0 }}
                                                    animate={{ width: `${lastPrediction.probability * 100}%` }}
                                                    transition={{ duration: 1, ease: "easeOut" }}
                                                    className={`h-full rounded-full ${lastPrediction.probability > 0.7 ? 'bg-gradient-to-r from-red-500 to-rose-600' :
                                                            lastPrediction.probability > 0.4 ? 'bg-gradient-to-r from-yellow-500 to-orange-600' :
                                                                'bg-gradient-to-r from-green-500 to-emerald-600'
                                                        }`}
                                                />
                                            </div>
                                        </div>

                                        {/* Prediction Result */}
                                        <motion.div
                                            initial={{ scale: 0 }}
                                            animate={{ scale: 1 }}
                                            transition={{ delay: 0.3, type: "spring" }}
                                            className={`p-4 rounded-xl text-center ${lastPrediction.prediction
                                                    ? 'bg-gradient-to-r from-red-500 to-rose-600'
                                                    : 'bg-gradient-to-r from-green-500 to-emerald-600'
                                                } text-white`}
                                        >
                                            <p className="text-2xl font-bold mb-2">
                                                {lastPrediction.prediction ? 'HIGH RISK' : 'LOW RISK'}
                                            </p>
                                            <p className="text-sm">
                                                {lastPrediction.prediction ? 'Employee Likely to Leave' : 'Employee Likely to Stay'}
                                            </p>
                                        </motion.div>

                                        {recs && recs.recommendations && (
                                            <div className="mt-6">
                                                <h3 className="text-lg font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-3">
                                                    Recommendations
                                                </h3>
                                                <div className="space-y-2">
                                                    {recs.recommendations.map((r, i) => (
                                                        <motion.div
                                                            key={i}
                                                            initial={{ opacity: 0, x: 20 }}
                                                            animate={{ opacity: 1, x: 0 }}
                                                            transition={{ delay: i * 0.05 }}
                                                            className="p-3 rounded-lg bg-white/70 dark:bg-gray-700/70 border border-indigo-200 dark:border-gray-600"
                                                        >
                                                            <div className="flex justify-between items-center">
                                                                <div className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                                                                    {r.action}
                                                                </div>
                                                                <div className="text-xs font-bold text-emerald-600">
                                                                    -{(r.delta * 100).toFixed(1)}%
                                                                </div>
                                                            </div>
                                                            <div className="text-xs text-gray-600 dark:text-gray-300 mt-1">
                                                                {r.rationale}
                                                            </div>
                                                        </motion.div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {lime && lime.explanation && (
                                            <div className="mt-6">
                                                <h3 className="text-lg font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-3">
                                                    LIME Explanation
                                                </h3>
                                                <div className="space-y-2">
                                                    {lime.explanation.slice(0, 10).map((p, i) => (
                                                        <motion.div
                                                            key={i}
                                                            initial={{ opacity: 0, x: 20 }}
                                                            animate={{ opacity: 1, x: 0 }}
                                                            transition={{ delay: i * 0.03 }}
                                                            className="p-3 rounded-lg bg-white/70 dark:bg-gray-700/70 border border-purple-200 dark:border-gray-600"
                                                        >
                                                            <div className="flex justify-between items-center">
                                                                <div className="text-xs font-semibold text-gray-800 dark:text-gray-200">
                                                                    {p.feature}
                                                                </div>
                                                                <div className={`text-xs font-bold ${p.weight >= 0 ? 'text-rose-600' : 'text-emerald-600'}`}>
                                                                    {p.weight.toFixed(3)}
                                                                </div>
                                                            </div>
                                                        </motion.div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </motion.div>
                                )}
                            </AnimatePresence>
                        </motion.div>

                        {/* SHAP Summary */}
                        <motion.div
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.3 }}
                        >
                            <ShapSummary />
                        </motion.div>
                    </div>
                </div>
            </div>

            {/* Sidebar */}
            <AnimatePresence>
                {sidebarOpen && (
                    <>
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
                            onClick={() => setSidebarOpen(false)}
                        />
                        <motion.aside
                            className="fixed right-0 top-0 h-full w-96 z-50 bg-white/90 backdrop-blur-xl shadow-2xl dark:bg-gray-800/90 p-6"
                            initial={{ x: 400 }}
                            animate={{ x: 0 }}
                            exit={{ x: 400 }}
                            transition={{ type: "spring", damping: 25 }}
                        >
                            <button
                                onClick={() => setSidebarOpen(false)}
                                className="absolute top-4 right-4 text-2xl hover:rotate-90 transition-transform"
                            >
                                ✕
                            </button>
                            <h4 className="text-2xl font-bold mb-4 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                                Model Insights
                            </h4>
                            <div className="space-y-4">
                                <p className="text-sm text-gray-600 dark:text-gray-300">
                                    Explore detailed metrics, confusion matrices, and SHAP feature importance to understand model predictions.
                                </p>
                                {bestModel && (
                                    <div className="p-4 bg-gradient-to-br from-indigo-100 to-purple-100 dark:from-gray-700 dark:to-gray-600 rounded-xl">
                                        <p className="text-sm font-semibold mb-2">Top Performer</p>
                                        <p className="text-xl font-bold">{bestModel.name}</p>
                                        <p className="text-sm mt-1">Accuracy: {(bestModel.auc * 100).toFixed(2)}%</p>
                                    </div>
                                )}
                            </div>
                        </motion.aside>
                    </>
                )}
            </AnimatePresence>
        </div>
    )
}

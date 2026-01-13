import React, { useState } from 'react';
import { motion } from 'framer-motion';

const defaultValues = {
    Age: 35,
    MonthlyIncome: 5000,
    OverTime: "No",
    JobRole: "Research Scientist",
    YearsAtCompany: 5,
    JobSatisfaction: 3,
    WorkLifeBalance: 2
};

export default function PredictForm({ onPredict, predicting }) {
    const [form, setForm] = useState(defaultValues);

    function handleChange(e) {
        const { name, value } = e.target;
        setForm(prev => ({ ...prev, [name]: value }));
    }

    function submit(e) {
        e.preventDefault();
        const payload = { ...form };
        payload.Age = Number(payload.Age);
        payload.MonthlyIncome = Number(payload.MonthlyIncome);
        payload.YearsAtCompany = Number(payload.YearsAtCompany);
        payload.JobSatisfaction = Number(payload.JobSatisfaction);
        payload.WorkLifeBalance = Number(payload.WorkLifeBalance);
        onPredict(payload);
    }

    const RatingInput = ({ label, name, value }) => (
        <div className="space-y-2">
            <label className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                {label}
            </label>
            <div className="flex gap-2">
                {[1, 2, 3, 4].map(rating => (
                    <motion.button
                        key={rating}
                        type="button"
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={() => setForm(prev => ({ ...prev, [name]: rating }))}
                        className={`flex-1 py-2 rounded-lg font-semibold transition-all ${
                            value === rating
                                ? 'bg-gradient-to-r from-indigo-500 to-purple-600 text-white shadow-lg scale-105'
                                : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200'
                        }`}
                    >
                        {rating}
                    </motion.button>
                ))}
            </div>
        </div>
    );

    return (
        <form onSubmit={submit} className="space-y-4">
            <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Age
                </label>
                <input
                    type="number"
                    name="Age"
                    value={form.Age}
                    onChange={handleChange}
                    min={18}
                    max={65}
                    className="w-full p-3 border-2 border-indigo-200 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100"
                />
            </div>

            <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Monthly Income
                </label>
                <input
                    type="number"
                    name="MonthlyIncome"
                    value={form.MonthlyIncome}
                    onChange={handleChange}
                    min={1000}
                    max={20000}
                    step={100}
                    className="w-full p-3 border-2 border-indigo-200 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100"
                />
            </div>

            <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Years at Company
                </label>
                <input
                    type="number"
                    name="YearsAtCompany"
                    value={form.YearsAtCompany}
                    onChange={handleChange}
                    min={0}
                    max={40}
                    className="w-full p-3 border-2 border-indigo-200 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100"
                />
            </div>

            <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Overtime
                </label>
                <div className="flex gap-2">
                    {['No', 'Yes'].map(option => (
                        <motion.button
                            key={option}
                            type="button"
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => setForm(prev => ({ ...prev, OverTime: option }))}
                            className={`flex-1 py-3 rounded-lg font-semibold transition-all ${
                                form.OverTime === option
                                    ? option === 'Yes'
                                        ? 'bg-gradient-to-r from-red-500 to-rose-600 text-white shadow-lg'
                                        : 'bg-gradient-to-r from-green-500 to-emerald-600 text-white shadow-lg'
                                    : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                            }`}
                        >
                            {option}
                        </motion.button>
                    ))}
                </div>
            </div>

            <div className="space-y-2">
                <label className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                    Job Role
                </label>
                <select
                    name="JobRole"
                    value={form.JobRole}
                    onChange={handleChange}
                    className="w-full p-3 border-2 border-indigo-200 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-all dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100"
                >
                    <option>Research Scientist</option>
                    <option>Sales Executive</option>
                    <option>Manager</option>
                    <option>Laboratory Technician</option>
                    <option>Manufacturing Director</option>
                    <option>Healthcare Representative</option>
                    <option>Research Director</option>
                    <option>Sales Representative</option>
                </select>
            </div>

            <RatingInput
                label="Job Satisfaction (1-4)"
                name="JobSatisfaction"
                value={form.JobSatisfaction}
            />

            <RatingInput
                label="Work-Life Balance (1-4)"
                name="WorkLifeBalance"
                value={form.WorkLifeBalance}
            />

            <motion.button
                type="submit"
                disabled={predicting}
                whileHover={{ scale: predicting ? 1 : 1.05 }}
                whileTap={{ scale: predicting ? 1 : 0.95 }}
                className={`w-full py-4 rounded-xl font-bold text-lg shadow-lg transition-all ${
                    predicting
                        ? 'bg-gray-400 cursor-not-allowed'
                        : 'bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-600 hover:shadow-2xl text-white'
                }`}
            >
                {predicting ? (
                    <span className="flex items-center justify-center gap-2">
                        <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                        Analyzing...
                    </span>
                ) : (
                    'Predict Attrition Risk'
                )}
            </motion.button>
        </form>
    )
}

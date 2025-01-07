import React, { useState } from 'react';
import { Thermometer, Hash } from 'lucide-react';
import { model } from "../../declarations/model";

const Slider = ({ value, onChange, min, max, step, label, icon: Icon }) => (
  <div className="space-y-2">
    <div className="flex items-center gap-2">
      {Icon && <Icon className="w-4 h-4 text-gray-400" />}
      <label className="text-sm text-gray-300">{label}: {value}</label>
    </div>
    <input
      type="range"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      min={min}
      max={max}
      step={step}
      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
    />
  </div>
);

const GPTInterface = () => {
  const [question, setQuestion] = useState('');
  const [tokens, setTokens] = useState(10);
  const [temperature, setTemperature] = useState(0.2);
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      // Convert values to ensure correct types
      const params = {
        prompt: String(question),
        numTokens: Math.floor(Number(tokens)), // Ensure integer
        temperature: Number(temperature)        // Ensure number
      };

      console.log('Sending request with params:', params);

      const response = await model.generate(
        params.prompt,
        params.numTokens,
        params.temperature
      );

      console.log('Received response:', response);

      if ('Ok' in response) {
        setResult(response.Ok);
      } else if ('Err' in response) {
        console.error('Error from model:', response.Err);
        setResult(`Error: ${response.Err}`);
      } else {
        console.error('Unexpected response structure:', response);
        setResult('Unexpected response format');
      }
    } catch (error) {
      console.error('Error details:', {
        name: error.name,
        message: error.message,
        stack: error.stack
      });
      setResult(`Error: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-2xl mx-auto space-y-8">
        {/* Logo Section */}
        <div className="flex justify-center">
          <img src="../assets/logo_white.svg" alt="Company Logo" className="h-16" />
        </div>

        {/* Main Card */}
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 shadow-xl">
          <h2 className="text-white text-center text-2xl font-bold mb-6">AI Text Generation</h2>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Question Input */}
            <div className="space-y-2">
              <label className="text-sm text-gray-300">Enter your question</label>
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 text-white rounded-lg p-2 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="What would you like to know?"
              />
            </div>

            <Slider
              value={tokens}
              onChange={setTokens}
              min={1}
              max={100}
              step={1}
              label="Number of tokens"
              icon={Hash}
            />

            <Slider
              value={temperature}
              onChange={setTemperature}
              min={0}
              max={1}
              step={0.1}
              label="Temperature"
              icon={Thermometer}
            />

            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
            >
              {isLoading ? 'Generating...' : 'Generate'}
            </button>
          </form>

          {/* Results Section */}
          {result && (
            <div className="mt-6 p-4 bg-gray-700 rounded-lg border border-gray-600">
              <h3 className="text-sm text-gray-300 mb-2 font-medium">Generated Response:</h3>
              <p className="text-white whitespace-pre-wrap">{result}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default GPTInterface;
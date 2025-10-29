import React, { useState, useEffect, useRef } from 'react';
import { Mail, Shield, AlertTriangle, CheckCircle, BarChart3, Brain, Zap, Sparkles, Lock, TrendingUp } from 'lucide-react';

// Animated Stat Component with counting animation
const AnimatedStat = ({ value, suffix }) => {
  const nodeRef = useRef(null);

  useEffect(() => {
    if (nodeRef.current) {
      let startValue = 0;
      const duration = 2000; // 2 seconds
      const startTime = Date.now();
      
      const updateValue = () => {
        const currentTime = Date.now();
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth animation
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const currentValue = startValue + (value - startValue) * easeOutQuart;
        
        const formattedValue = value % 1 === 0 ? Math.floor(currentValue) : currentValue.toFixed(1);
        nodeRef.current.textContent = `${formattedValue}${suffix || ''}`;
        
        if (progress < 1) {
          requestAnimationFrame(updateValue);
        }
      };
      
      updateValue();
    }
  }, [value, suffix]);

  return <span ref={nodeRef}>0{suffix || ''}</span>;
};

const SpamDetectorWebsite = () => {
  const [emailText, setEmailText] = useState('');
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedModel, setSelectedModel] = useState('naive_bayes');
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [particles, setParticles] = useState([]);

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  useEffect(() => {
    const newParticles = Array.from({ length: 30 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 4 + 2,
      duration: Math.random() * 20 + 10,
      delay: Math.random() * 5
    }));
    setParticles(newParticles);
  }, []);

  // API call to Python backend
  const handleAnalyze = async () => {
    if (!emailText.trim()) return;

    setIsAnalyzing(true);
    setResult(null);
    
    const API_URL = 'https://spam-detector-api-50o3.onrender.com';

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          emailText: emailText,
          model: selectedModel 
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data.analysis);

    } catch (error) {
      console.error('Failed to analyze email:', error);
      setResult({
        isSpam: true,
        confidence: 100,
        error: 'Failed to connect to AI server. Is the backend running on port 5000?',
        features: {
          keywordCount: 0,
          exclamationMarks: 0,
          capsWords: 0,
          textLength: emailText.length
        }
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const sampleEmails = {
    spam: "CONGRATULATIONS! You've WON a FREE iPhone 15! Click here NOW to claim your prize! Limited time offer! Act fast!",
    ham: "Hi John, just wanted to follow up on our meeting yesterday. Can you send me the project report by Friday? Thanks!"
  };

  const loadSample = (type) => {
    setEmailText(sampleEmails[type]);
    setResult(null);
  };

  return (
    <div className="min-h-screen bg-black text-white overflow-hidden relative">
      <style jsx>{`
        @keyframes float {
          0%, 100% { transform: translateY(0px) translateX(0px); }
          25% { transform: translateY(-20px) translateX(10px); }
          50% { transform: translateY(-40px) translateX(-10px); }
          75% { transform: translateY(-20px) translateX(10px); }
        }
        .animate-float {
          animation: float linear infinite;
        }
        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(30px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-slideUp {
          animation: slideUp 0.6s ease-out forwards;
        }
        @keyframes scaleIn {
          from {
            opacity: 0;
            transform: scale(0.9);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
        .animate-scaleIn {
          animation: scaleIn 0.5s ease-out forwards;
        }
        @keyframes glow {
          0%, 100% { box-shadow: 0 0 20px rgba(147,51,234,0.5); }
          50% { box-shadow: 0 0 40px rgba(147,51,234,0.8), 0 0 60px rgba(59,130,246,0.5); }
        }
        .animate-glow {
          animation: glow 2s ease-in-out infinite;
        }
        .glass-effect {
          background: rgba(255, 255, 255, 0.05);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .glass-effect-strong {
          background: rgba(255, 255, 255, 0.08);
          backdrop-filter: blur(30px);
          -webkit-backdrop-filter: blur(30px);
          border: 1px solid rgba(255, 255, 255, 0.15);
        }
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .shimmer::after {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
          animation: shimmer 2s infinite;
        }
      `}</style>

      {/* Animated Background */}
      <div className="fixed inset-0 bg-gradient-to-br from-purple-900/20 via-black to-blue-900/20" />
      
      {/* Floating Particles */}
      {particles.map(particle => (
        <div
          key={particle.id}
          className="absolute rounded-full bg-white/10 blur-sm animate-float"
          style={{
            left: `${particle.x}%`,
            top: `${particle.y}%`,
            width: `${particle.size}px`,
            height: `${particle.size}px`,
            animationDuration: `${particle.duration}s`,
            animationDelay: `${particle.delay}s`
          }}
        />
      ))}

      {/* Mouse Glow Effect */}
      <div 
        className="fixed w-96 h-96 rounded-full pointer-events-none transition-all duration-300 opacity-30 blur-3xl"
        style={{
          background: 'radial-gradient(circle, rgba(147,51,234,0.3) 0%, transparent 70%)',
          left: mousePosition.x - 192,
          top: mousePosition.y - 192
        }}
      />

      {/* Header */}
      <header className="relative z-10 glass-effect border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between animate-slideUp">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-blue-600 rounded-2xl blur-xl opacity-50 animate-glow" />
                <div className="relative bg-gradient-to-r from-purple-600 to-blue-600 p-3 rounded-2xl">
                  <Shield className="w-8 h-8 text-white" />
                </div>
              </div>
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-white via-purple-200 to-blue-200 bg-clip-text text-transparent">
                  Spam Detector AI
                </h1>
                <p className="text-sm text-gray-400 flex items-center gap-2 mt-1">
                  <Sparkles className="w-4 h-4" />
                  Powered by Machine Learning
                </p>
              </div>
            </div>
            <div className="glass-effect px-6 py-3 rounded-2xl flex items-center space-x-3 transform hover:scale-105 transition-all duration-300">
              <Brain className="w-5 h-5 text-purple-400" />
              <span className="text-sm font-medium text-gray-200">
                {selectedModel === 'naive_bayes' ? 'Naive Bayes' : 'SVM'} Model
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Input */}
          <div className="lg:col-span-2 space-y-6">
            {/* Input Card */}
            <div className="glass-effect-strong rounded-3xl p-8 transform hover:scale-[1.02] transition-all duration-500 animate-slideUp">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                  Analyze Email
                </h2>
                <div className="flex space-x-3">
                  <button
                    onClick={() => loadSample('spam')}
                    className="glass-effect px-5 py-2.5 text-sm text-red-300 rounded-xl hover:bg-red-500/20 transition-all duration-300 transform hover:scale-105 border border-red-500/30"
                  >
                    Load Spam
                  </button>
                  <button
                    onClick={() => loadSample('ham')}
                    className="glass-effect px-5 py-2.5 text-sm text-green-300 rounded-xl hover:bg-green-500/20 transition-all duration-300 transform hover:scale-105 border border-green-500/30"
                  >
                    Load Ham
                  </button>
                </div>
              </div>

              <div className="relative group">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-blue-600/20 rounded-2xl blur-xl group-hover:blur-2xl transition-all duration-300 opacity-0 group-hover:opacity-100" />
                <textarea
                  value={emailText}
                  onChange={(e) => setEmailText(e.target.value)}
                  placeholder="Paste your email content here..."
                  className="relative w-full h-64 p-6 glass-effect rounded-2xl focus:ring-2 focus:ring-purple-500/50 transition-all resize-none text-white placeholder-gray-500 border border-white/10 focus:border-purple-500/50"
                />
              </div>

              <div className="mt-6 flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="glass-effect px-6 py-3 rounded-xl focus:ring-2 focus:ring-purple-500/50 transition-all text-white border border-white/10 cursor-pointer"
                  >
                    <option value="naive_bayes" className="bg-gray-900">Naive Bayes</option>
                    <option value="svm" className="bg-gray-900">Support Vector Machine</option>
                  </select>
                  <span className="text-sm text-gray-400 glass-effect px-4 py-2 rounded-lg">
                    {emailText.length} characters
                  </span>
                </div>
                <button
                  onClick={handleAnalyze}
                  disabled={!emailText.trim() || isAnalyzing}
                  className="relative group px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 flex items-center space-x-2 overflow-hidden"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-blue-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  {isAnalyzing ? (
                    <>
                      <div className="relative w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span className="relative">Analyzing...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="relative w-5 h-5" />
                      <span className="relative">Analyze</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Results Card */}
            {result && (
              <div className="glass-effect-strong rounded-3xl p-8 animate-scaleIn transform hover:scale-[1.02] transition-all duration-500">
                {result.error ? (
                  <div className="text-center">
                    <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />
                    <h2 className="text-xl font-bold text-red-400 mb-2">Connection Error</h2>
                    <p className="text-gray-400">{result.error}</p>
                  </div>
                ) : (
                  <>
                    <div className="flex items-center justify-between mb-6">
                      <h2 className="text-2xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
                        Analysis Results
                      </h2>
                      {result.isSpam ? (
                        <div className="flex items-center space-x-3 glass-effect px-5 py-3 rounded-xl border border-red-500/30 bg-red-500/10">
                          <AlertTriangle className="w-6 h-6 text-red-400" />
                          <span className="font-semibold text-red-300">SPAM DETECTED</span>
                        </div>
                      ) : (
                        <div className="flex items-center space-x-3 glass-effect px-5 py-3 rounded-xl border border-green-500/30 bg-green-500/10">
                          <CheckCircle className="w-6 h-6 text-green-400" />
                          <span className="font-semibold text-green-300">LEGITIMATE</span>
                        </div>
                      )}
                    </div>

                    {/* Confidence Bar */}
                    <div className="mb-8">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-sm font-medium text-gray-300">Confidence Score</span>
                        <span className="text-3xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                          {result.confidence.toFixed(1)}%
                        </span>
                      </div>
                      <div className="relative w-full h-3 glass-effect rounded-full overflow-hidden">
                        <div className="absolute inset-0 bg-gradient-to-r from-white/5 to-white/10" />
                        <div
                          className={`relative h-full rounded-full transition-all duration-1000 ${
                            result.isSpam 
                              ? 'bg-gradient-to-r from-red-500 via-red-400 to-pink-500' 
                              : 'bg-gradient-to-r from-green-500 via-emerald-400 to-teal-500'
                          }`}
                          style={{ width: `${result.confidence}%` }}
                        >
                          <div className="absolute inset-0 bg-gradient-to-r from-white/30 to-transparent animate-pulse" />
                        </div>
                      </div>
                    </div>

                    {/* Features */}
                    <div className="grid grid-cols-2 gap-4 mb-6">
                      {[
                        { label: 'Spam Keywords', value: result.features.keywordCount, icon: Lock },
                        { label: 'Exclamation Marks', value: result.features.exclamationMarks, icon: AlertTriangle },
                        { label: 'CAPS Words', value: result.features.capsWords, icon: TrendingUp },
                        { label: 'Text Length', value: result.features.textLength, icon: Mail }
                      ].map((item, idx) => (
                        <div 
                          key={idx}
                          className="glass-effect p-5 rounded-2xl transform hover:scale-105 transition-all duration-300 border border-white/10 group"
                          style={{ animationDelay: `${idx * 0.1}s` }}
                        >
                          <div className="flex items-center justify-between mb-2">
                            <div className="text-sm text-gray-400">{item.label}</div>
                            <item.icon className="w-4 h-4 text-purple-400 opacity-50 group-hover:opacity-100 transition-opacity" />
                          </div>
                          <div className="text-3xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                            {item.value}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Found Keywords */}
                    {result.foundKeywords && result.foundKeywords.length > 0 && (
                      <div className="animate-slideUp">
                        <h3 className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
                          <Sparkles className="w-4 h-4 text-purple-400" />
                          Detected Spam Keywords
                        </h3>
                        <div className="flex flex-wrap gap-3">
                          {result.foundKeywords.map((keyword, idx) => (
                            <span 
                              key={idx} 
                              className="glass-effect px-4 py-2 text-red-300 text-sm rounded-xl border border-red-500/30 transform hover:scale-110 transition-all duration-300 hover:bg-red-500/20"
                              style={{ animationDelay: `${idx * 0.05}s` }}
                            >
                              {keyword}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>

          {/* Right Column - Info */}
          <div className="space-y-6">
            {/* Stats Card */}
            <div className="relative glass-effect-strong rounded-3xl p-8 overflow-hidden transform hover:scale-105 transition-all duration-500 animate-slideUp border border-white/10">
              <div className="absolute top-0 right-0 w-40 h-40 bg-gradient-to-br from-purple-600/30 to-blue-600/30 rounded-full blur-3xl" />
              <div className="relative">
                <div className="flex items-center space-x-3 mb-6">
                  <BarChart3 className="w-8 h-8 text-purple-400" />
                  <h3 className="text-xl font-bold text-white">Model Stats</h3>
                </div>
                <div className="space-y-6">
                  {[
                    { label: 'Accuracy', value: 98.7, suffix: '%' },
                    { label: 'Precision', value: 97.2, suffix: '%' },
                    { label: 'Training Dataset', value: 5574, sub: 'SMS messages' }
                  ].map((stat, idx) => (
                    <div key={idx} className="transform hover:translate-x-2 transition-all duration-300">
                      <div className="text-sm text-gray-400 mb-1">{stat.label}</div>
                      <div className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                        <AnimatedStat value={stat.value} suffix={stat.suffix || ''} />
                      </div>
                      {stat.sub && <div className="text-sm text-gray-500 mt-1">{stat.sub}</div>}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* How It Works */}
            <div className="glass-effect-strong rounded-3xl p-8 transform hover:scale-105 transition-all duration-500 border border-white/10">
              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                <Brain className="w-6 h-6 text-purple-400" />
                How It Works
              </h3>
              <div className="space-y-5">
                {[
                  { num: 1, title: 'Text Preprocessing', desc: 'Cleaning and tokenizing email content', color: 'from-purple-500 to-purple-600' },
                  { num: 2, title: 'Feature Extraction', desc: 'TF-IDF vectorization of text', color: 'from-blue-500 to-blue-600' },
                  { num: 3, title: 'ML Classification', desc: 'Naive Bayes or SVM prediction', color: 'from-green-500 to-green-600' }
                ].map((step, idx) => (
                  <div key={idx} className="flex space-x-4 group">
                    <div className={`flex-shrink-0 w-10 h-10 bg-gradient-to-r ${step.color} rounded-xl flex items-center justify-center font-bold text-white shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                      {step.num}
                    </div>
                    <div className="flex-1">
                      <div className="font-semibold text-white mb-1">{step.title}</div>
                      <div className="text-sm text-gray-400">{step.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Tips */}
            <div className="glass-effect-strong rounded-3xl p-8 border border-amber-500/20 bg-amber-500/5 transform hover:scale-105 transition-all duration-500">
              <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                <Lock className="w-6 h-6 text-amber-400" />
                Spam Indicators
              </h3>
              <ul className="space-y-3">
                {[
                  'Urgent action requests',
                  'Too good to be true offers',
                  'Excessive punctuation (!!!)',
                  'ALL CAPS WORDS',
                  'Suspicious links'
                ].map((tip, idx) => (
                  <li key={idx} className="flex items-start text-sm text-gray-300 transform hover:translate-x-2 transition-all duration-300">
                    <span className="text-amber-400 mr-3 text-lg">•</span>
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      // ... Footer ...
      <footer className="relative z-10 glass-effect border-t border-white/10 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-400">
            
            {/* Main Tagline */}
            <p className="text-sm mb-2">
              AI-Driven Classification Engine • Naive Bayes & SVM Algorithms
            </p>

            {/* YOUR NAME ADDED HERE: Bright White for visibility */}
            <p className="text-sm text-white font-bold mb-3">
              Project developed by: Prajwal Kamte
            </p>
            
            {/* Copyright */}
            <p className="text-xs flex items-center justify-center gap-2">
              <Shield className="w-4 h-4" />
              © 2025 Spam Detector AI. Protecting your inbox with AI.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};



export default SpamDetectorWebsite;
import React, { useState, useEffect } from 'react';
import { MapPin, Search, TrendingUp, AlertTriangle, RefreshCw, BarChart3, Globe, Clock, Activity, Zap, Waves, Navigation, Layers, MessageSquare, Brain, Sparkles, AlertCircle, ChevronRight, Flame, Target, Info } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar, Cell, PieChart, Pie, Legend, CartesianGrid } from 'recharts';
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://earthquake-backend.onrender.com/api'  // <-- Your Render URL here
  : 'http://localhost:8000/api';
  const App = () => {
  const [stats, setStats] = useState(null);
  const [hotspots, setHotspots] = useState([]);
  const [recentEarthquakes, setRecentEarthquakes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searching, setSearching] = useState(false);
  const [activeTab, setActiveTab] = useState('search');
  const [magnitudeDistributionData, setMagnitudeDistributionData] = useState([]);
  const [depthAnalysisData, setDepthAnalysisData] = useState([]);
  const [ragQuestion, setRagQuestion] = useState('');
  const [ragAnswer, setRagAnswer] = useState(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [extractedEntities, setExtractedEntities] = useState([]);
  const [systemStatus, setSystemStatus] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [summaryDashboardData, setSummaryDashboardData] = useState([]);
  const [featureInputs, setFeatureInputs] = useState({
    magnitude: '',
    depth: '',
    latitude: '',
    longitude: '',
    delta_t: '',
    log_cum_energy_50: ''
  });
  const [numPredictions, setNumPredictions] = useState(1);
  const [predicting, setPredicting] = useState(false);
  const [forecastResult, setForecastResult] = useState(null);
  const [forecastError, setForecastError] = useState(null);
  useEffect(() => {
    fetchData();
    checkSystemHealth();
  
    const interval = setInterval(() => {
      fetchData();
      checkSystemHealth();
    }, 300000);
  
    return () => clearInterval(interval);
  }, []);
  const checkSystemHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) throw new Error('Health check failed');
      const data = await response.json();
      setSystemStatus(data);
      setApiError(null);
    } catch (error) {
      console.error('Error checking health:', error);
      setApiError('API connection failed. Please ensure backend is running on http://localhost:8000');
    }
  };
  const fetchData = async () => {
    setLoading(true);
    setApiError(null);
  
    try {
      const [statsRes, hotspotsRes, recentRes] = await Promise.all([
        fetch(`${API_BASE_URL}/stats`).catch(() => null),
        fetch(`${API_BASE_URL}/hotspots?limit=10`).catch(() => null),
        fetch(`${API_BASE_URL}/earthquakes/recent?limit=20`).catch(() => null)
      ]);
      if (!statsRes && !hotspotsRes) {
        throw new Error('All API requests failed');
      }
      if (statsRes?.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      
        if (statsData.category_distribution) {
          const magData = Object.entries(statsData.category_distribution).map(([category, count]) => ({
            category,
            count: count,
            percentage: ((count / statsData.total_count) * 100).toFixed(1)
          }));
          setMagnitudeDistributionData(magData);
        }
        if (statsData.depth_distribution) {
          const depthData = Object.entries(statsData.depth_distribution).map(([category, count]) => ({
            category,
            count: count,
            percentage: ((count / statsData.total_count) * 100).toFixed(1)
          }));
          setDepthAnalysisData(depthData);
        }
        const dashboardMetrics = [
          {
            name: 'Total Events',
            value: statsData.total_count || 0,
            category: 'normal',
            color: '#10b981'
          },
          {
            name: 'Events/Day',
            value: statsData.time_metrics?.events_per_day ? parseFloat(statsData.time_metrics.events_per_day.toFixed(1)) : 0,
            category: 'normal',
            color: '#06b6d4'
          },
          {
            name: 'Active Regions',
            value: statsData.geographic_metrics?.unique_regions || 0,
            category: 'moderate',
            color: '#f59e0b'
          },
          {
            name: 'High Risk Events',
            value: statsData.risk_metrics?.high_risk_count || 0,
            category: 'moderate',
            color: '#f97316'
          },
          {
            name: 'Tsunami Warnings',
            value: statsData.risk_metrics?.tsunami_warnings || 0,
            category: 'critical',
            color: '#ef4444'
          },
          {
            name: 'Max Magnitude',
            value: statsData.magnitude_stats?.max ? parseFloat(statsData.magnitude_stats.max.toFixed(1)) : 0,
            category: 'critical',
            color: '#dc2626'
          }
        ];
      
        setSummaryDashboardData(dashboardMetrics);
      }
      if (hotspotsRes?.ok) {
        const hotspotsData = await hotspotsRes.json();
        setHotspots(hotspotsData.hotspots || []);
      }
      if (recentRes?.ok) {
        const recentData = await recentRes.json();
        setRecentEarthquakes(recentData.earthquakes || []);
      }
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching data:', error);
      setApiError('Failed to fetch data. Please check your backend connection.');
    } finally {
      setLoading(false);
    }
  };
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearching(true);
    setApiError(null);
  
    try {
      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, top_k: 5 })
      });
    
      if (!response.ok) throw new Error('Search failed');
    
      const data = await response.json();
      setSearchResults(data.results || []);
      setExtractedEntities(data.extracted_entities || []);
    } catch (error) {
      console.error('Error searching:', error);
      setApiError('Search failed. Please try again.');
    } finally {
      setSearching(false);
    }
  };
  const handleRagQuery = async () => {
    if (!ragQuestion.trim()) return;
    setRagLoading(true);
    setApiError(null);
  
    try {
      const response = await fetch(`${API_BASE_URL}/rag/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: ragQuestion })
      });
    
      if (!response.ok) throw new Error('RAG query failed');
    
      const data = await response.json();
      setRagAnswer(data);
    } catch (error) {
      console.error('Error querying RAG:', error);
      setRagAnswer({
        error: 'Failed to process query',
        answer: 'Unable to connect to the AI system. Please ensure the backend server is running.'
      });
    } finally {
      setRagLoading(false);
    }
  };
  const handleInputChange = (field, value) => {
    setFeatureInputs(prev => ({
      ...prev,
      [field]: value
    }));
    setForecastError(null);
  };
  const validateInputs = () => {
    const { magnitude, depth, latitude, longitude, delta_t, log_cum_energy_50 } = featureInputs;
  
    if (!magnitude || !depth || !latitude || !longitude || !delta_t || !log_cum_energy_50) {
      setForecastError('All fields are required');
      return false;
    }
    const mag = parseFloat(magnitude);
    const dep = parseFloat(depth);
    const lat = parseFloat(latitude);
    const lon = parseFloat(longitude);
    const dt = parseFloat(delta_t);
    const energy = parseFloat(log_cum_energy_50);
    if (isNaN(mag) || isNaN(dep) || isNaN(lat) || isNaN(lon) || isNaN(dt) || isNaN(energy)) {
      setForecastError('All inputs must be valid numbers');
      return false;
    }
    if (mag < 0 || mag > 10) {
      setForecastError('Magnitude must be between 0 and 10');
      return false;
    }
    if (dep < 0) {
      setForecastError('Depth must be positive');
      return false;
    }
    if (lat < -90 || lat > 90) {
      setForecastError('Latitude must be between -90 and 90');
      return false;
    }
    if (lon < -180 || lon > 180) {
      setForecastError('Longitude must be between -180 and 180');
      return false;
    }
    if (dt < 0) {
      setForecastError('Delta time must be positive');
      return false;
    }
    return true;
  };
  const handleForecastPredict = async () => {
    if (!validateInputs()) return;
    setPredicting(true);
    setForecastError(null);
    setForecastResult(null);
    try {
      const response = await fetch(`${API_BASE_URL}/predict/lstm`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_predictions: numPredictions,
          latitude: parseFloat(featureInputs.latitude),
          longitude: parseFloat(featureInputs.longitude),
          depth: parseFloat(featureInputs.depth),
          magnitude: parseFloat(featureInputs.magnitude),
          delta_t: parseFloat(featureInputs.delta_t),
          log_cum_energy_50: parseFloat(featureInputs.log_cum_energy_50)
        })
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Prediction failed');
      }
      const data = await response.json();
      setForecastResult(data);
    } catch (err) {
      console.error('Prediction error:', err);
      setForecastError(err.message || 'Failed to get prediction. Ensure backend is running.');
    } finally {
      setPredicting(false);
    }
  };
  const loadExample = (exampleType) => {
    const examples = {
      typical: {
        magnitude: '4.5',
        depth: '50',
        latitude: '35.68',
        longitude: '139.76',
        delta_t: '3600',
        log_cum_energy_50: '12.5'
      },
      shallow: {
        magnitude: '5.2',
        depth: '10',
        latitude: '37.77',
        longitude: '-122.42',
        delta_t: '7200',
        log_cum_energy_50: '13.2'
      },
      deep: {
        magnitude: '6.0',
        depth: '150',
        latitude: '-6.21',
        longitude: '106.85',
        delta_t: '1800',
        log_cum_energy_50: '14.5'
      }
    };
    setFeatureInputs(examples[exampleType]);
    setForecastError(null);
    setForecastResult(null);
  };
  const resetForecastForm = () => {
    setFeatureInputs({
      magnitude: '',
      depth: '',
      latitude: '',
      longitude: '',
      delta_t: '',
      log_cum_energy_50: ''
    });
    setForecastError(null);
    setForecastResult(null);
  };
  const handleRefresh = async () => {
    setLoading(true);
    try {
      const refreshRes = await fetch(`${API_BASE_URL}/refresh`, { method: 'POST' });
      if (refreshRes.ok) {
        await fetchData();
        await checkSystemHealth();
      } else {
        throw new Error('Refresh failed');
      }
    } catch (error) {
      console.error('Error refreshing:', error);
      setApiError('Failed to refresh data from USGS.');
      setLoading(false);
    }
  };
  const getRiskColor = (magnitude) => {
    if (!magnitude) return 'text-slate-600';
    if (magnitude >= 7) return 'text-rose-600';
    if (magnitude >= 6) return 'text-orange-500';
    if (magnitude >= 5) return 'text-amber-500';
    return 'text-emerald-500';
  };
  const getRiskBgColor = (riskLevel) => {
    if (riskLevel === 'High') return 'bg-rose-50 text-rose-700 border-rose-200';
    if (riskLevel === 'Moderate') return 'bg-amber-50 text-amber-700 border-amber-200';
    return 'bg-emerald-50 text-emerald-700 border-emerald-200';
  };
  const getMagnitudeColor = (magnitude) => {
    if (!magnitude) return '#94a3b8';
    if (magnitude >= 7) return '#dc2626';
    if (magnitude >= 6) return '#f97316';
    if (magnitude >= 5) return '#eab308';
    return '#22c55e';
  };
  const getEntityColor = (label) => {
    const colors = {
      'GPE': 'bg-sky-50 text-sky-700 border-sky-200',
      'LOC': 'bg-emerald-50 text-emerald-700 border-emerald-200',
      'DATE': 'bg-purple-50 text-purple-700 border-purple-200',
      'TIME': 'bg-pink-50 text-pink-700 border-pink-200',
      'CARDINAL': 'bg-amber-50 text-amber-700 border-amber-200'
    };
    return colors[label] || 'bg-slate-50 text-slate-700 border-slate-200';
  };
  const CustomBarTooltip = ({ active, payload }) => {
    if (active && payload?.length) {
      return (
        <div className="bg-white px-4 py-3 rounded-xl shadow-lg border border-slate-200">
          <p className="text-sm font-semibold text-slate-900">{payload[0].payload.name}</p>
          <p className="text-sm text-sky-600 font-medium mt-1">Value: {payload[0].value}</p>
          <p className="text-xs text-slate-500 mt-1 capitalize">{payload[0].payload.category} Risk</p>
        </div>
      );
    }
    return null;
  };
  const CustomDistributionTooltip = ({ active, payload }) => {
    if (active && payload?.length) {
      return (
        <div className="bg-white px-4 py-3 rounded-xl shadow-lg border border-slate-200">
          <p className="text-sm font-semibold text-slate-900">{payload[0].payload.category}</p>
          <p className="text-sm text-sky-600 font-medium mt-1">Count: {payload[0].value}</p>
          <p className="text-xs text-slate-500 mt-1">{payload[0].payload.percentage}% of total</p>
        </div>
      );
    }
    return null;
  };
  const COLORS = {
    'Minor': '#10b981',
    'Moderate': '#fbbf24',
    'Strong': '#f97316',
    'Major': '#ef4444',
    'Great': '#dc2626',
    'Shallow': '#06b6d4',
    'Intermediate': '#8b5cf6',
    'Deep': '#ec4899'
  };
  const fields = [
    {
      id: 'magnitude',
      label: 'Current Magnitude',
      icon: Zap,
      placeholder: '4.5',
      description: 'Magnitude of the current earthquake (0-10)',
      unit: 'M',
      color: 'text-sky-500'
    },
    {
      id: 'depth',
      label: 'Depth',
      icon: Layers,
      placeholder: '50',
      description: 'Depth of the earthquake epicenter',
      unit: 'km',
      color: 'text-purple-500'
    },
    {
      id: 'latitude',
      label: 'Latitude',
      icon: Navigation,
      placeholder: '35.68',
      description: 'Latitude coordinate (-90 to 90)',
      unit: '°',
      color: 'text-emerald-500'
    },
    {
      id: 'longitude',
      label: 'Longitude',
      icon: Navigation,
      placeholder: '139.76',
      description: 'Longitude coordinate (-180 to 180)',
      unit: '°',
      color: 'text-blue-500'
    },
    {
      id: 'delta_t',
      label: 'Time Delta',
      icon: Clock,
      placeholder: '3600',
      description: 'Time since previous earthquake in seconds',
      unit: 'sec',
      color: 'text-amber-500'
    },
    {
      id: 'log_cum_energy_50',
      label: 'Log Cumulative Energy',
      icon: Activity,
      placeholder: '12.5',
      description: 'Log10 of cumulative seismic energy (last 50 events)',
      unit: 'log10',
      color: 'text-rose-500'
    }
  ];
  if (loading && !stats) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-sky-50 via-blue-50 to-indigo-50 flex items-center justify-center">
        <div className="text-center">
          <div className="relative inline-block">
            <div className="w-24 h-24 rounded-full border-4 border-sky-100 border-t-sky-500 animate-spin"></div>
            <Globe className="w-10 h-10 text-sky-500 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2" />
          </div>
          <p className="text-2xl text-slate-800 font-bold mt-6 mb-2">Loading Earthquake Data</p>
          <p className="text-sm text-slate-500">Connecting to USGS systems</p>
        </div>
      </div>
    );
  }
  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-5">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <div className="w-14 h-14 bg-gradient-to-br from-sky-400 to-blue-500 rounded-2xl flex items-center justify-center shadow-lg">
                  <Globe className="w-8 h-8 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-emerald-400 rounded-full border-2 border-white"></div>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-800">Earthquake Intelligence</h1>
                <p className="text-sm text-slate-500 mt-0.5">Real-time USGS seismic monitoring</p>
              </div>
            </div>
          
            <div className="flex items-center space-x-3">
              {systemStatus && (
                <div className="flex items-center space-x-2 px-4 py-2 bg-slate-50 rounded-xl border border-slate-200">
                  <div className={`w-2 h-2 rounded-full ${systemStatus.rag_ready ? 'bg-emerald-500' : 'bg-amber-500'} animate-pulse`}></div>
                  <span className="text-xs font-medium text-slate-600">System Active</span>
                </div>
              )}
            
              <button
                onClick={handleRefresh}
                disabled={loading}
                className="flex items-center space-x-2 bg-sky-500 text-white px-5 py-2.5 rounded-xl hover:bg-sky-600 transition-all shadow-sm hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                <span className="text-sm font-medium">Refresh</span>
              </button>
            </div>
          </div>
          {apiError && (
            <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-xl flex items-start space-x-2">
              <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-amber-800">{apiError}</p>
            </div>
          )}
        </div>
      </div>
      {/* Stats Overview */}
      {stats && (
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5 hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <p className="text-sm text-slate-500 mb-1">Total Events</p>
                  <p className="text-3xl font-bold text-slate-800">{stats.total_count || 0}</p>
                </div>
                <div className="w-12 h-12 bg-sky-50 rounded-xl flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-sky-500" />
                </div>
              </div>
            </div>
          
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5 hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <p className="text-sm text-slate-500 mb-1">Avg Magnitude</p>
                  <p className="text-3xl font-bold text-emerald-600">
                    {stats.magnitude_stats?.mean ? stats.magnitude_stats.mean.toFixed(1) : 'N/A'}
                  </p>
                </div>
                <div className="w-12 h-12 bg-emerald-50 rounded-xl flex items-center justify-center">
                  <Activity className="w-6 h-6 text-emerald-500" />
                </div>
              </div>
            </div>
          
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5 hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <p className="text-sm text-slate-500 mb-1">Max Magnitude</p>
                  <p className="text-3xl font-bold text-rose-600">
                    {stats.magnitude_stats?.max ? stats.magnitude_stats.max.toFixed(1) : 'N/A'}
                  </p>
                </div>
                <div className="w-12 h-12 bg-rose-50 rounded-xl flex items-center justify-center">
                  <AlertTriangle className="w-6 h-6 text-rose-500" />
                </div>
              </div>
            </div>
          
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-5 hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <p className="text-sm text-slate-500 mb-1">Active Regions</p>
                  <p className="text-3xl font-bold text-blue-600">
                    {stats.geographic_metrics?.unique_regions || 0}
                  </p>
                </div>
                <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center">
                  <MapPin className="w-6 h-6 text-blue-500" />
                </div>
              </div>
            </div>
          </div>
          {/* Additional Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
            <div className="bg-gradient-to-br from-sky-500 to-blue-600 rounded-2xl shadow-sm p-5 text-white">
              <div className="flex items-center justify-between mb-3">
                <Zap className="w-8 h-8 opacity-80" />
                <span className="text-xs bg-white/20 px-3 py-1 rounded-full">Daily</span>
              </div>
              <p className="text-sm opacity-90 mb-1">Events per Day</p>
              <p className="text-3xl font-bold">
                {stats.time_metrics?.events_per_day ? stats.time_metrics.events_per_day.toFixed(1) : '0'}
              </p>
            </div>
            <div className="bg-gradient-to-br from-orange-500 to-rose-600 rounded-2xl shadow-sm p-5 text-white">
              <div className="flex items-center justify-between mb-3">
                <Flame className="w-8 h-8 opacity-80" />
                <span className="text-xs bg-white/20 px-3 py-1 rounded-full">M ≥ 6.0</span>
              </div>
              <p className="text-sm opacity-90 mb-1">High Risk Events</p>
              <p className="text-3xl font-bold">{stats.risk_metrics?.high_risk_count || 0}</p>
            </div>
            <div className="bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl shadow-sm p-5 text-white">
              <div className="flex items-center justify-between mb-3">
                <Waves className="w-8 h-8 opacity-80" />
                <span className="text-xs bg-white/20 px-3 py-1 rounded-full">Alert</span>
              </div>
              <p className="text-sm opacity-90 mb-1">Tsunami Warnings</p>
              <p className="text-3xl font-bold">{stats.risk_metrics?.tsunami_warnings || 0}</p>
            </div>
          </div>
        </div>
      )}
      {/* Charts Section */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-bold text-slate-800 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2 text-sky-500" />
                Magnitude Distribution
              </h2>
            </div>
            {magnitudeDistributionData?.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={magnitudeDistributionData}>
                  <XAxis dataKey="category" stroke="#64748b" style={{ fontSize: '12px' }} />
                  <YAxis stroke="#64748b" style={{ fontSize: '12px' }} />
                  <Tooltip content={<CustomDistributionTooltip />} />
                  <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                    {magnitudeDistributionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[entry.category] || '#0ea5e9'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-64 flex items-center justify-center text-slate-400">
                <div className="text-center">
                  <BarChart3 className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No data available</p>
                </div>
              </div>
            )}
          </div>
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-bold text-slate-800 flex items-center">
                <Layers className="w-5 h-5 mr-2 text-sky-500" />
                Depth Analysis
              </h2>
            </div>
            {depthAnalysisData?.length > 0 ? (
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={depthAnalysisData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ category, percentage }) => `${category}: ${percentage}%`}
                    outerRadius={95}
                    dataKey="count"
                  >
                    {depthAnalysisData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[entry.category] || '#0ea5e9'} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomDistributionTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-64 flex items-center justify-center text-slate-400">
                <div className="text-center">
                  <Layers className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No data available</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      {/* Tabs */}
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-2">
          <div className="flex space-x-2 overflow-x-auto">
            {[
              { id: 'rag', label: 'AI Assistant', icon: Brain },
              { id: 'search', label: 'Search', icon: Search },
              { id: 'hotspots', label: 'Hotspots', icon: Flame },
              { id: 'forecast', label: 'LSTM Forecast', icon: TrendingUp }
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-5 py-3 rounded-xl font-medium transition-all whitespace-nowrap flex items-center space-x-2 ${
                    activeTab === tab.id
                      ? 'bg-sky-500 text-white shadow-sm'
                      : 'text-slate-600 hover:bg-slate-50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
      {/* Tab Content */}
      <div className="max-w-7xl mx-auto px-6 py-6 pb-12">
        {activeTab === 'rag' && (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200">
            <div className="p-6 border-b border-slate-200">
              <div className="flex items-center space-x-3 mb-2">
                <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-800">AI Question & Answer</h2>
                  <p className="text-sm text-slate-500">Ask questions in natural language</p>
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="flex space-x-3 mb-6">
                <input
                  type="text"
                  value={ragQuestion}
                  onChange={(e) => setRagQuestion(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleRagQuery()}
                  placeholder="Ask anything about earthquakes..."
                  className="flex-1 px-5 py-3.5 border border-slate-300 rounded-xl focus:ring-2 focus:ring-sky-500 focus:border-transparent text-base"
                />
                <button
                  onClick={handleRagQuery}
                  disabled={ragLoading}
                  className="px-8 py-3.5 bg-sky-500 text-white rounded-xl hover:bg-sky-600 transition-all flex items-center disabled:opacity-50 font-medium shadow-sm"
                >
                  {ragLoading ? <RefreshCw className="w-5 h-5 mr-2 animate-spin" /> : <MessageSquare className="w-5 h-5 mr-2" />}
                  Ask
                </button>
              </div>
              {ragAnswer && (
                <div className="mt-6 p-6 bg-slate-50 rounded-xl border border-slate-200">
                  <div className="flex items-start space-x-3">
                    <div className="w-8 h-8 bg-sky-100 rounded-lg flex items-center justify-center flex-shrink-0">
                      <Brain className="w-5 h-5 text-sky-600" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-base font-semibold text-slate-800 mb-3">Response</h3>
                      <p className="text-slate-700 leading-relaxed whitespace-pre-wrap">{ragAnswer.answer}</p>
                    
                      {ragAnswer.extracted_entities?.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-slate-200">
                          <p className="text-sm font-semibold text-slate-700 mb-2">Extracted Entities:</p>
                          <div className="flex flex-wrap gap-2">
                            {ragAnswer.extracted_entities.map((entity, idx) => (
                              <span key={idx} className={`px-3 py-1 rounded-lg text-xs font-medium border ${getEntityColor(entity.label)}`}>
                                {entity.text} <span className="opacity-60">({entity.label})</span>
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    
                      {ragAnswer.sources?.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-slate-200">
                          <p className="text-sm font-semibold text-slate-700 mb-2">Sources:</p>
                          <div className="space-y-2">
                            {ragAnswer.sources.map((src, idx) => (
                              <div key={idx} className="text-xs bg-white p-3 rounded-lg border border-slate-200">
                                <span className="font-semibold">M{src.magnitude}</span> - {src.place}
                                <span className="text-slate-500 ml-2">({src.timestamp})</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}
              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-5 bg-purple-50 rounded-xl border border-purple-200">
                  <h4 className="font-semibold text-slate-800 mb-3 flex items-center">
                    <MessageSquare className="w-4 h-4 mr-2 text-purple-600" />
                    Example Questions
                  </h4>
                  <ul className="space-y-2 text-sm text-slate-700">
                    {[
                      "What were the strongest earthquakes in the Pacific region?",
                      "Tell me about recent deep earthquakes",
                      "Which regions had the most seismic activity?",
                      "Summarize major earthquake events this month"
                    ].map((q, i) => (
                      <li
                        key={i}
                        className="cursor-pointer hover:text-purple-600 hover:bg-white p-2 rounded-lg transition flex items-start"
                        onClick={() => setRagQuestion(q)}
                      >
                        <ChevronRight className="w-4 h-4 mr-1 mt-0.5 flex-shrink-0" />
                        <span>{q}</span>
                      </li>
                    ))}
                  </ul>
                </div>
                <div className="p-5 bg-sky-50 rounded-xl border border-sky-200">
                  <h4 className="font-semibold text-slate-800 mb-3 flex items-center">
                    <Brain className="w-4 h-4 mr-2 text-sky-600" />
                    How It Works
                  </h4>
                  <ul className="space-y-3 text-sm text-slate-700">
                    {[
                      "Analyze your question",
                      "Search relevant data",
                      "Generate context",
                      "Provide the relevant answer"
                    ].map((step, i) => (
                      <li key={i} className="flex items-start bg-white p-2 rounded-lg border border-sky-100">
                        <span className="text-sky-600 font-bold mr-2 flex-shrink-0">{i + 1}</span>
                        <span>{step}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          </div>
        )}
        {activeTab === 'search' && (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200">
            <div className="p-6 border-b border-slate-200">
              <div className="flex items-center space-x-3 mb-2">
                <div className="w-10 h-10 bg-gradient-to-br from-sky-500 to-blue-500 rounded-xl flex items-center justify-center">
                  <Search className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-800">Semantic Search</h2>
                  <p className="text-sm text-slate-500">Find earthquakes using natural language</p>
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="flex space-x-3 mb-6">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  placeholder="e.g., 'large earthquakes in Pacific Ring of Fire'"
                  className="flex-1 px-5 py-3.5 border border-slate-300 rounded-xl focus:ring-2 focus:ring-sky-500 focus:border-transparent text-base"
                />
                <button
                  onClick={handleSearch}
                  disabled={searching}
                  className="px-8 py-3.5 bg-sky-500 text-white rounded-xl hover:bg-sky-600 transition-all flex items-center disabled:opacity-50 font-medium shadow-sm"
                >
                  {searching ? <RefreshCw className="w-5 h-5 mr-2 animate-spin" /> : <Search className="w-5 h-5 mr-2" />}
                  Search
                </button>
              </div>
              {extractedEntities?.length > 0 && (
                <div className="mb-6 p-4 bg-sky-50 rounded-xl border border-sky-200">
                  <p className="text-sm font-semibold text-slate-700 mb-3 flex items-center">
                    <Sparkles className="w-4 h-4 mr-2 text-sky-600" />
                    Extracted Entities
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {extractedEntities.map((entity, idx) => (
                      <span key={idx} className={`px-3 py-1.5 rounded-lg text-xs font-medium border ${getEntityColor(entity.label)}`}>
                        {entity.text} <span className="opacity-60">({entity.label})</span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
              <div className="space-y-4">
                {searchResults?.length > 0 ? (
                  searchResults.map((result, idx) => (
                    <div key={result.id || idx} className="p-5 bg-slate-50 rounded-xl border border-slate-200 hover:shadow-md hover:border-sky-300 transition-all">
                      <div className="flex justify-between items-start mb-3">
                        <h3 className="font-bold text-lg text-slate-800">{result.title || 'Earthquake Event'}</h3>
                        <span className="px-3 py-1 bg-sky-100 text-sky-700 text-xs font-bold rounded-lg border border-sky-200">
                          {result.similarity_score ? (result.similarity_score * 100).toFixed(1) : 'N/A'}% match
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 mb-3 flex items-center">
                        <MapPin className="w-4 h-4 mr-1.5 text-slate-400" />
                        {result.place || 'Unknown location'}
                      </p>
                      <div className="flex flex-wrap gap-3 text-sm">
                        <span className={`font-bold ${getRiskColor(result.magnitude)} flex items-center`}>
                          <Zap className="w-4 h-4 mr-1" />
                          M {result.magnitude ? result.magnitude.toFixed(1) : 'N/A'}
                        </span>
                        <span className="text-slate-700 flex items-center">
                          <Layers className="w-4 h-4 mr-1 text-slate-400" />
                          {result.depth ? result.depth.toFixed(1) : 'N/A'} km
                        </span>
                        <span className="px-3 py-1 bg-white rounded-lg font-medium border border-slate-200">
                          {result.magnitude_category || 'Unknown'}
                        </span>
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-16 text-slate-400">
                    <Search className="w-16 h-16 mx-auto mb-4 opacity-50" />
                    <p className="text-lg font-medium">Start searching for earthquakes</p>
                    <p className="text-sm mt-2">Try: "strong earthquakes near Japan" or "deep earthquakes magnitude 5+"</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
        {activeTab === 'hotspots' && (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200">
            <div className="p-6 border-b border-slate-200">
              <div className="flex items-center space-x-3 mb-2">
                <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-rose-500 rounded-xl flex items-center justify-center">
                  <Flame className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-800">Earthquake Hotspots</h2>
                  <p className="text-sm text-slate-500">Regions with highest seismic activity</p>
                </div>
              </div>
            </div>
            <div className="p-6">
              {hotspots?.length > 0 ? (
                <div className="space-y-4">
                  {hotspots.map((hotspot, idx) => (
                    <div key={idx} className="border border-slate-200 rounded-2xl overflow-hidden hover:shadow-md hover:border-sky-300 transition-all">
                      <div className="flex items-center justify-between p-6 bg-slate-50">
                        <div className="flex items-center space-x-4 flex-1">
                          <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-br from-sky-500 to-blue-600 text-white rounded-xl font-bold text-lg shadow-sm">
                            {idx + 1}
                          </div>
                          <div className="flex-1">
                            <h3 className="font-bold text-slate-800 text-lg">{hotspot.region || 'Unknown'}</h3>
                            <p className="text-sm text-slate-500 mt-0.5">{hotspot.count || 0} recorded events</p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-4">
                          <div className="text-right">
                            <p className="text-xs text-slate-500">Avg Magnitude</p>
                            <p className={`text-2xl font-bold ${getRiskColor(hotspot.avg_magnitude)}`}>
                              {hotspot.avg_magnitude || 'N/A'}
                            </p>
                          </div>
                        </div>
                      </div>
                      <div className="p-6 bg-white">
                        <div className="flex items-center mb-4">
                          <Clock className="w-5 h-5 text-sky-500 mr-2" />
                          <h4 className="text-sm font-bold text-slate-700">Latest Event</h4>
                        </div>
                      
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                          <div className="space-y-3">
                            <div className="flex items-start p-3 bg-slate-50 rounded-lg border border-slate-200">
                              <Clock className="w-5 h-5 text-slate-400 mr-3 mt-0.5 flex-shrink-0" />
                              <div className="flex-1 min-w-0">
                                <p className="text-xs text-slate-500">Time</p>
                                <p className="text-sm font-semibold text-slate-800 truncate">
                                  {hotspot.latest_earthquake?.time || 'N/A'}
                                </p>
                                <p className="text-xs text-sky-600 mt-1 font-medium">
                                  {hotspot.latest_earthquake?.time_ago || 'Unknown'}
                                </p>
                              </div>
                            </div>
                            <div className="flex items-start p-3 bg-slate-50 rounded-lg border border-slate-200">
                              <Navigation className="w-5 h-5 text-slate-400 mr-3 mt-0.5 flex-shrink-0" />
                              <div className="flex-1">
                                <p className="text-xs text-slate-500">Coordinates</p>
                                <p className="text-sm font-semibold text-slate-800">
                                  {hotspot.latest_earthquake?.latitude?.toFixed(3)}°, {hotspot.latest_earthquake?.longitude?.toFixed(3)}°
                                </p>
                              </div>
                            </div>
                          </div>
                          <div className="space-y-3">
                            <div className="flex items-start p-3 bg-slate-50 rounded-lg border border-slate-200">
                              <Zap className="w-5 h-5 text-slate-400 mr-3 mt-0.5 flex-shrink-0" />
                              <div className="flex-1">
                                <p className="text-xs text-slate-500">Magnitude</p>
                                <div className="flex items-center space-x-2 mt-1">
                                  <span className={`text-3xl font-bold ${getRiskColor(hotspot.latest_earthquake?.magnitude)}`}>
                                    M{hotspot.latest_earthquake?.magnitude || 'N/A'}
                                  </span>
                                </div>
                              </div>
                            </div>
                            <div className="flex items-start p-3 bg-slate-50 rounded-lg border border-slate-200">
                              <Layers className="w-5 h-5 text-slate-400 mr-3 mt-0.5 flex-shrink-0" />
                              <div className="flex-1">
                                <p className="text-xs text-slate-500">Depth</p>
                                <p className="text-sm font-bold text-slate-800 mt-1">
                                  {hotspot.latest_earthquake?.depth || 'N/A'} km
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>
                        <div className="grid grid-cols-3 gap-3 mb-4">
                          {[
                            { label: 'Max Magnitude', value: hotspot.max_magnitude, color: getRiskColor(hotspot.max_magnitude) },
                            { label: 'Avg Depth', value: `${hotspot.avg_depth || 'N/A'} km`, color: 'text-slate-800' },
                            { label: 'Total Events', value: hotspot.count || 0, color: 'text-sky-600' }
                          ].map((stat, i) => (
                            <div key={i} className="text-center p-4 bg-slate-50 rounded-xl border border-slate-200">
                              <p className="text-xs text-slate-500 mb-1">{stat.label}</p>
                              <p className={`text-lg font-bold ${stat.color}`}>{stat.value}</p>
                            </div>
                          ))}
                        </div>
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs text-slate-500 font-medium">Activity Level</span>
                            <span className="text-xs font-bold text-slate-700">
                              {((hotspot.count / (hotspots[0]?.count || 1)) * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-200 rounded-full h-2.5">
                            <div
                              className="bg-gradient-to-r from-sky-500 to-blue-600 h-2.5 rounded-full transition-all duration-1000"
                              style={{ width: `${hotspots[0]?.count ? (hotspot.count / hotspots[0].count) * 100 : 0}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-16 text-slate-400">
                  <Flame className="w-20 h-20 mx-auto mb-4 opacity-50" />
                  <p className="text-xl font-medium">No hotspot data available</p>
                  <p className="text-sm mt-2">Refresh data to load earthquake hotspots</p>
                </div>
              )}
            </div>
          </div>
        )}
        {activeTab === 'forecast' && (
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200">
            <div className="p-6 border-b border-slate-200">
              <div className="flex items-center space-x-3 mb-2">
                <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-xl flex items-center justify-center">
                  <TrendingUp className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-slate-800">LSTM Earthquake Forecast</h2>
                  <p className="text-sm text-slate-500">Feature-based magnitude prediction using deep learning</p>
                </div>
              </div>
            </div>
            <div className="p-6">
              <div className="bg-sky-50 border border-sky-200 rounded-xl p-4 mb-6 flex items-start space-x-3">
                <Info className="w-5 h-5 text-sky-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm text-sky-800 font-medium">How it works</p>
                  <p className="text-sm text-sky-700 mt-1">
                    The LSTM model analyzes 6 seismic features to predict the magnitude of the next earthquake.
                    Enter current earthquake data and temporal patterns to get AI-powered forecasts.
                  </p>
                </div>
              </div>
              <div className="bg-white rounded-xl border border-slate-200 p-5 mb-6">
                <h3 className="text-base font-bold text-slate-800 mb-4">Quick Examples</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {[
                    { type: 'typical', label: 'Typical Event', desc: 'Moderate shallow earthquake' },
                    { type: 'shallow', label: 'Shallow Strong', desc: 'High magnitude, shallow depth' },
                    { type: 'deep', label: 'Deep Event', desc: 'Deep earthquake with high energy' }
                  ].map((example) => (
                    <button
                      key={example.type}
                      onClick={() => loadExample(example.type)}
                      className="p-4 bg-gradient-to-br from-slate-50 to-sky-50 rounded-xl border border-slate-200 hover:border-sky-300 hover:shadow-md transition-all text-left"
                    >
                      <p className="font-semibold text-slate-800 text-sm">{example.label}</p>
                      <p className="text-xs text-slate-500 mt-1">{example.desc}</p>
                    </button>
                  ))}
                </div>
              </div>
              <div className="max-w-md mb-6">
                <label className="block text-sm font-semibold text-slate-700 mb-2">Number of Predictions (1-10)</label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={numPredictions}
                  onChange={(e) => setNumPredictions(Math.min(10, Math.max(1, parseInt(e.target.value) || 1)))}
                  className="w-full px-4 py-3 border border-slate-300 rounded-xl focus:ring-2 focus:ring-sky-500 focus:border-transparent"
                />
              </div>
              <div className="bg-slate-50 rounded-xl border border-slate-200 p-6 mb-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-base font-bold text-slate-800">Seismic Feature Inputs</h3>
                  <button
                    onClick={resetForecastForm}
                    className="flex items-center space-x-2 px-3 py-1.5 bg-white text-slate-700 rounded-lg hover:bg-slate-100 transition-all border border-slate-200 text-sm"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Reset</span>
                  </button>
                </div>
              
                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  {fields.map((field) => {
                    const Icon = field.icon;
                    return (
                      <div key={field.id} className="space-y-2">
                        <label className="flex items-center space-x-2 text-sm font-semibold text-slate-700">
                          <Icon className={`w-4 h-4 ${field.color}`} />
                          <span>{field.label}</span>
                        </label>
                        <div className="relative">
                          <input
                            type="number"
                            step="any"
                            value={featureInputs[field.id]}
                            onChange={(e) => handleInputChange(field.id, e.target.value)}
                            placeholder={field.placeholder}
                            className="w-full px-4 py-3 pr-16 border border-slate-300 rounded-xl focus:ring-2 focus:ring-sky-500 focus:border-transparent text-base bg-white"
                          />
                          <span className="absolute right-4 top-1/2 transform -translate-y-1/2 text-sm text-slate-400 font-medium">
                            {field.unit}
                          </span>
                        </div>
                        <p className="text-xs text-slate-500">{field.description}</p>
                      </div>
                    );
                  })}
                </div>
                {forecastError && (
                  <div className="mt-5 p-4 bg-rose-50 border border-rose-200 rounded-xl flex items-start space-x-2">
                    <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
                    <p className="text-sm text-rose-800">{forecastError}</p>
                  </div>
                )}
                <button
                  onClick={handleForecastPredict}
                  disabled={predicting}
                  className="mt-6 w-full md:w-auto px-10 py-4 bg-gradient-to-r from-sky-500 to-blue-600 text-white rounded-xl hover:from-sky-600 hover:to-blue-700 transition-all flex items-center justify-center disabled:opacity-50 font-medium shadow-lg hover:shadow-xl"
                >
                  {predicting ? (
                    <>
                      <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                      Analyzing Features...
                    </>
                  ) : (
                    <>
                      <TrendingUp className="w-5 h-5 mr-2" />
                      Predict Next Magnitude
                    </>
                  )}
                </button>
              </div>
              {forecastResult && (
                <div className="bg-white rounded-xl border border-slate-200 p-6 mb-6">
                  <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center">
                    <Target className="w-6 h-6 mr-2 text-sky-500" />
                    Prediction Results
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="bg-gradient-to-br from-slate-50 to-sky-50 p-6 rounded-xl border border-slate-200">
                      <p className="text-sm text-slate-500 mb-2">Predicted Magnitude</p>
                      <p className={`text-4xl font-bold ${getRiskColor(forecastResult.summary?.average_magnitude)}`}>
                        M {forecastResult.summary?.average_magnitude?.toFixed(2) || 'N/A'}
                      </p>
                    </div>
                  
                    <div className="bg-gradient-to-br from-slate-50 to-sky-50 p-6 rounded-xl border border-slate-200">
                      <p className="text-sm text-slate-500 mb-2">Risk Assessment</p>
                      <span className={`inline-block px-4 py-2 rounded-xl text-lg font-bold border ${getRiskBgColor(forecastResult.summary?.risk_assessment)}`}>
                        {forecastResult.summary?.risk_assessment || 'Unknown'}
                      </span>
                    </div>
                    <div className="bg-gradient-to-br from-slate-50 to-sky-50 p-6 rounded-xl border border-slate-200">
                      <p className="text-sm text-slate-500 mb-2">Confidence</p>
                      <p className="text-2xl font-bold text-sky-600 capitalize">
                        {forecastResult.predictions?.[0]?.confidence || 'N/A'}
                      </p>
                    </div>
                  </div>
                  {forecastResult.predictions && forecastResult.predictions.length > 0 && (
                    <div className="space-y-3 mb-6">
                      <h4 className="text-sm font-bold text-slate-700 mb-3">Detailed Forecast</h4>
                      {forecastResult.predictions.map((pred, idx) => (
                        <div key={idx} className="p-5 bg-slate-50 rounded-xl border border-slate-200 flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <div className="w-12 h-12 bg-sky-100 rounded-xl flex items-center justify-center">
                              <TrendingUp className="w-6 h-6 text-sky-600" />
                            </div>
                            <div>
                              <p className="text-sm text-slate-500">Next Earthquake</p>
                              <p className={`text-3xl font-bold ${getRiskColor(pred.predicted_magnitude)}`}>
                                M {pred.predicted_magnitude?.toFixed(2)}
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <span className="px-3 py-1.5 bg-sky-100 text-sky-700 text-xs font-semibold rounded-lg capitalize">
                              {pred.confidence} confidence
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="p-5 bg-slate-50 rounded-xl border border-slate-200 mb-6">
                    <h4 className="text-sm font-bold text-slate-700 mb-3">Input Features Used</h4>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                      {fields.map((field) => (
                        <div key={field.id} className="flex items-center space-x-2">
                          <span className="text-slate-500">{field.label}:</span>
                          <span className="font-semibold text-slate-800">
                            {featureInputs[field.id]} {field.unit}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                  {forecastResult.disclaimer && (
                    <div className="p-4 bg-amber-50 border border-amber-200 rounded-xl flex items-start space-x-2">
                      <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                      <p className="text-xs text-amber-800">{forecastResult.disclaimer}</p>
                    </div>
                  )}
                </div>
              )}
              <div className="bg-white rounded-xl border border-slate-200 p-6">
                <h3 className="text-base font-bold text-slate-800 mb-4">Feature Definitions</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {fields.map((field) => {
                    const Icon = field.icon;
                    return (
                      <div key={field.id} className="p-4 bg-slate-50 rounded-xl border border-slate-200">
                        <div className="flex items-start space-x-3">
                          <div className={`w-8 h-8 rounded-lg flex items-center justify-center bg-white border border-slate-200 flex-shrink-0`}>
                            <Icon className={`w-4 h-4 ${field.color}`} />
                          </div>
                          <div className="flex-1">
                            <p className="font-semibold text-slate-800 text-sm">{field.label}</p>
                            <p className="text-xs text-slate-600 mt-1">{field.description}</p>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
export default App;
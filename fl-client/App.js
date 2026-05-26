import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, ScrollView, TextInput, SafeAreaView } from 'react-native';

export default function App() {
  const [serverId, setServerId] = useState('100.87.17.13'); // Replace with your host Tailscale IP
  const [clientId] = useState(`client-${Math.random().toString(36).substring(7)}`);
  const [status, setStatus] = useState('Disconnected');
  const [logs, setLogs] = useState([]);
  const [currentRound, setCurrentRound] = useState(0);
  const ws = useRef(null);

  const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prevLogs) => [`[${timestamp}] ${message}`, ...prevLogs]);
  };

  const connectToServer = () => {
    if (!serverId) return;
    setStatus('Connecting...');
    
    // Clean up the input string
    let cleanIp = serverId.replace('http://', '').replace('https://', '').trim();
    if (!cleanIp.includes(':')) {
      cleanIp = `${cleanIp}:8000`;
    }

    addLog(`Opening live WebSocket stream to: ws://${cleanIp}/ws/fl`);
    
    // Establish a persistent WebSocket connection to the FastAPI server
    ws.current = new WebSocket(`ws://${cleanIp}/ws/fl`);

    // Handle successful connection handshake
    ws.current.onopen = () => {
      setStatus('Connected');
      addLog(`SUCCESS: WebSocket streaming active. ID: ${clientId}`);
    };

    // Listen for live messages broadcasted from the server
    ws.current.onmessage = (e) => {
      const response = JSON.parse(e.data);
      if (response.type === 'ROUND_COMPLETE') {
        setCurrentRound(response.round);
        addLog(`🔥 Global Round ${response.round} finalized! Synchronized master weights.`);
      }
    };

    // Catch connection drops or timeout issues
    ws.current.onerror = (err) => {
      setStatus('Error');
      addLog(`WebSocket connection dropped or timed out.`);
    };

    // Handle socket closure
    ws.current.onclose = () => {
      setStatus('Disconnected');
      addLog('Socket connection closed by host.');
    };
  };

  const simulateLocalTrainingAndUpload = () => {
    if (status !== 'Connected') {
      addLog('Cannot execute: WebSocket stream is offline.');
      return;
    }
    
    addLog('Loading local dataset samples into memory...');
    
    setTimeout(() => {
      addLog('Training localized Tiny CNN over 5 epochs...');
      
      // Generating our localized weight matrix delta parameters
      const fakeWeights = {
        "conv1.weight": Array(72).fill(0).map(() => Math.random() * 0.1),
        "conv1.bias": Array(8).fill(0).map(() => Math.random() * 0.01),
        "conv2.weight": Array(1152).fill(0).map(() => Math.random() * 0.1),
        "conv2.bias": Array(16).fill(0).map(() => Math.random() * 0.01),
        "fc1.weight": Array(4000).fill(0).map(() => Math.random() * 0.1),
        "fc1.bias": Array(10).fill(0).map(() => Math.random() * 0.01)
      };
      
      addLog('Streaming weight matrices over open WebSocket channel...');
      
      // Instantly push the data over the open socket connection frame
      ws.current.send(JSON.stringify({
        type: 'CLIENT_UPDATE',
        client_id: clientId,
        weights: fakeWeights
      }));
      
    }, 1500); // 1.5-second training processing simulation delay
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Federated Edge Client</Text>
        <Text style={[styles.badge, status === 'Connected' ? styles.online : styles.offline]}>{status}</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.label}>Server Tailscale Target IP</Text>
        <TextInput 
          style={styles.input} 
          value={serverId} 
          onChangeText={setServerId} 
          placeholder="100.x.x.x"
          keyboardType="numeric"
        />
        <TouchableOpacity style={styles.button} onPress={connectToServer}>
          <Text style={styles.buttonText}>Establish Connection</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.statsContainer}>
        <View style={styles.statBox}>
          <Text style={styles.statVal}>{currentRound}</Text>
          <Text style={styles.statLabel}>Global Round</Text>
        </View>
        <View style={styles.statBox}>
          <Text style={styles.statVal}>FEMNIST</Text>
          <Text style={styles.statLabel}>Dataset Context</Text>
        </View>
      </View>

      <TouchableOpacity style={[styles.actionBtn, status !== 'Connected' && styles.disabled]} onPress={simulateLocalTrainingAndUpload} disabled={status !== 'Connected'}>
        <Text style={styles.actionBtnText}>Execute Local Training Round</Text>
      </TouchableOpacity>

      <Text style={styles.logTitle}>System Execution Logs</Text>
      <ScrollView style={styles.logContainer}>
        {logs.map((log, index) => (
          <Text key={index} style={styles.logText}>{log}</Text>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F3F4F6', paddingHorizontal: 20, paddingTop: 40 },
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 },
  title: { fontSize: 22, fontWeight: '700', color: '#111827' },
  badge: { paddingHorizontal: 12, paddingY: 6, borderRadius: 20, fontSize: 12, fontWeight: '600', overflow: 'hidden' },
  online: { backgroundColor: '#D1FAE5', color: '#065F46' },
  offline: { backgroundColor: '#FEE2E2', color: '#991B1B' },
  card: { backgroundColor: '#FFF', padding: 16, borderRadius: 12, elevation: 2, marginBottom: 20 },
  label: { fontSize: 14, color: '#4B5563', marginBottom: 6, fontWeight: '500' },
  input: { borderWidth: 1, borderColor: '#D1D5DB', padding: 10, borderRadius: 8, marginBottom: 12, fontSize: 16 },
  button: { backgroundColor: '#2563EB', padding: 12, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: '#FFF', fontWeight: '600', fontSize: 16 },
  statsContainer: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 20 },
  statBox: { backgroundColor: '#FFF', padding: 16, borderRadius: 12, width: '48%', alignItems: 'center', elevation: 1 },
  statVal: { fontSize: 20, fontWeight: '700', color: '#1F2937' },
  statLabel: { fontSize: 12, color: '#6B7280', marginTop: 4 },
  actionBtn: { backgroundColor: '#10B981', padding: 16, borderRadius: 12, alignItems: 'center', marginBottom: 20 },
  disabled: { backgroundColor: '#A7F3D0', opacity: 0.6 },
  actionBtnText: { color: '#FFF', fontSize: 18, fontWeight: '700' },
  logTitle: { fontSize: 16, fontWeight: '600', color: '#374151', marginBottom: 8 },
  logContainer: { backgroundColor: '#1E293B', padding: 12, borderRadius: 12, flex: 1 },
  logText: { color: '#38BDF8', fontFamily: 'monospace', fontSize: 11, marginBottom: 4 }
});
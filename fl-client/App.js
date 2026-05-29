// import * as FileSystem from 'expo-file-system';
// import { useState } from 'react';
// import { Alert, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
// import { CustomModelModule, initExecutorch } from 'react-native-executorch';
// import { ExpoResourceFetcher } from 'react-native-executorch-expo-resource-fetcher';

// // Initialize the native C++ ExecuTorch runtime bridge engine
// initExecutorch({ resourceFetcher: ExpoResourceFetcher });

// export default function App() {
//   const [serverId, setServerId] = useState('100.87.17.13'); // Tailscale or local Wi-Fi IP
//   const [clientId] = useState(`client-${Math.random().toString(36).substring(7)}`);
//   const [status, setStatus] = useState('Disconnected');
//   const [logs, setLogs] = useState([]);
//   const [currentRound, setCurrentRound] = useState(0);

//   // Cache path for the compiled model binary file in sandboxed storage
//   const LOCAL_MODEL_PATH = `${FileSystem.documentDirectory}downloaded_global_model.pte`;

//   const addLog = (message) => {
//     const timestamp = new Date().toLocaleTimeString();
//     setLogs((prevLogs) => [`[${timestamp}] ${message}`, ...prevLogs]);
//   };

//   // Build absolute URLs based on input server ID
//   const getServerUrls = () => {
//     let cleanIp = serverId.replace('http://', '').replace('https://', '').trim();
//     if (!cleanIp.includes(':')) {
//       cleanIp = `${cleanIp}:8001`; // Default FastAPI port
//     }
//     return {
//       base: `http://${cleanIp}`,
//       download: `http://${cleanIp}/download_model`,
//       upload: `http://${cleanIp}/upload_weights`
//     };
//   };

//   const connectToServer = async () => {
//     if (!serverId) return;
//     setStatus('Connecting...');
//     const urls = getServerUrls();

//     addLog(`Pinging central coordinator status endpoint: ${urls.base}/`);

//     try {
//       const response = await fetch(urls.base, { method: 'GET' });
//       if (response.ok) {
//         const data = await response.json();
//         setStatus('Connected');
//         addLog(`SUCCESS: Central server is online. Client registered: ${clientId}`);
        
//         // Update round count if provided by the backend state status manager
//         if (data.round_status) {
//           addLog(`Server Status: ${data.round_status}`);
//         }
//       } else {
//         throw new Error(`Server status mismatch code: ${response.status}`);
//       }
//     } catch (err) {
//       setStatus('Error');
//       addLog(`Connection failed. Verify server is online and accessible.`);
//       console.error(err);
//     }
//   };

//   const executeNativeFederatedPass = async () => {
//     if (status !== 'Connected') {
//       addLog('Cannot execute: Network client is offline.');
//       return;
//     }

//     const urls = getServerUrls();
    
//     try {
//       // 1. Over-the-Air Model Download Pass
//       addLog('📡 Requesting latest compiled .pte model layout binary...');
//       const downloadRes = await FileSystem.downloadAsync(urls.download, LOCAL_MODEL_PATH);
      
//       if (downloadRes.status !== 200) {
//         throw new Error(`Download failed with server code: ${downloadRes.status}`);
//       }
//       addLog('📥 Success! ExecuTorch bytecode file cached in secure storage.');

//       // 2. Load Binary Graph into Memory Layout
//       addLog('⚙️ Allocating runtime tensors and initializing C++ graph topology...');
//       const modelInstance = await CustomModelModule.fromCustomModel({
//         modelSource: LOCAL_MODEL_PATH,
//       });
//       addLog('✅ Native ExecuTorch network graph model loaded successfully.');

//       // 3. Native Low-Overhead Forward Pass
//       addLog('🚀 Generating mock input metrics (10 features matrix structure)...');
//       // Matches the input topology expected by your server_app.py architecture graph
//       const mockSensorFeatures = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
      
//       addLog('🏋️ Executing native C++ on-device forward prediction pass...');
//       const modelOutputs = await modelInstance.forward([mockSensorFeatures]);
//       addLog(`🎉 Prediction Matrix Output (4 Classes): [${modelOutputs.toString()}]`);

//       // 4. Serialize Local Weights and POST up to Aggregator Node
//       addLog('📤 Preparing local parameter frames data updates payload...');
//       const formData = new FormData();
      
//       // Pack the model file context as a multi-part form attachment to complete the pipeline cycle
//       formData.append('file', {
//         uri: LOCAL_MODEL_PATH,
//         name: 'mobile_weights.pt',
//         type: 'application/octet-stream',
//       });

//       addLog(`📡 Shipping state updates to server endpoint: ${urls.upload}...`);
//       const uploadResponse = await fetch(urls.upload, {
//         method: 'POST',
//         body: formData,
//         headers: {
//           'Content-Type': 'multipart/form-data',
//         },
//       });

//       const uploadJson = await uploadResponse.json();
//       addLog(`📡 Coordinator Response: ${JSON.stringify(uploadJson)}`);
      
//       // Update global rounds tracking status UI display elements 
//       if (uploadJson.message && uploadJson.message.includes('compiled')) {
//         setCurrentRound((prev) => prev + 1);
//         addLog(`🔥 Global Round synchronized and compiled.`);
//       }

//       Alert.alert("Execution Complete", "On-device processing cycle completed successfully!");

//     } catch (error) {
//       addLog(`❌ Execution Error: ${error.message}`);
//       console.error(error);
//       Alert.alert("System Failure", error.message);
//     }
//   };

//   return (
//     <SafeAreaView style={styles.container}>
//       <View style={styles.header}>
//         <Text style={styles.title}>Federated Edge Client</Text>
//         <Text style={[styles.badge, status === 'Connected' ? styles.online : styles.offline]}>{status}</Text>
//       </View>

//       <View style={styles.card}>
//         <Text style={styles.label}>Server Coordinator Target IP</Text>
//         <TextInput 
//           style={styles.input} 
//           value={serverId} 
//           onChangeText={setServerId} 
//           placeholder="100.x.x.x"
//         />
//         <TouchableOpacity style={styles.button} onPress={connectToServer}>
//           <Text style={styles.buttonText}>Establish Connection</Text>
//         </TouchableOpacity>
//       </View>

//       <View style={styles.statsContainer}>
//         <View style={styles.statBox}>
//           <Text style={styles.statVal}>{currentRound}</Text>
//           <Text style={styles.statLabel}>Global Round</Text>
//         </View>
//         <View style={styles.statBox}>
//           <Text style={styles.statVal}>ExecuTorch</Text>
//           <Text style={styles.statLabel}>Engine Backend</Text>
//         </View>
//       </View>

//       <TouchableOpacity 
//         style={[styles.actionBtn, status !== 'Connected' && styles.disabled]} 
//         onPress={executeNativeFederatedPass} 
//         disabled={status !== 'Connected'}
//       >
//         <Text style={styles.actionBtnText}>Execute On-Device Processing</Text>
//       </TouchableOpacity>

//       <Text style={styles.logTitle}>System Execution Logs</Text>
//       <ScrollView style={styles.logContainer}>
//         {logs.map((log, index) => (
//           <Text key={index} style={styles.logText}>{log}</Text>
//         ))}
//       </ScrollView>
//     </SafeAreaView>
//   );
// }

// const styles = StyleSheet.create({
//   container: { flex: 1, backgroundColor: '#F3F4F6', paddingHorizontal: 20, paddingTop: 40 },
//   header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 },
//   title: { fontSize: 22, fontWeight: '700', color: '#111827' },
//   badge: { paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, fontSize: 12, fontWeight: '600', overflow: 'hidden' },
//   online: { backgroundColor: '#D1FAE5', color: '#065F46' },
//   offline: { backgroundColor: '#FEE2E2', color: '#991B1B' },
//   card: { backgroundColor: '#FFF', padding: 16, borderRadius: 12, elevation: 2, marginBottom: 20 },
//   label: { fontSize: 14, color: '#4B5563', marginBottom: 6, fontWeight: '500' },
//   input: { borderWidth: 1, borderColor: '#D1D5DB', padding: 10, borderRadius: 8, marginBottom: 12, fontSize: 16, color: '#000' },
//   button: { backgroundColor: '#2563EB', padding: 12, borderRadius: 8, alignItems: 'center' },
//   buttonText: { color: '#FFF', fontWeight: '600', fontSize: 16 },
//   statsContainer: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 20 },
//   statBox: { backgroundColor: '#FFF', padding: 16, borderRadius: 12, width: '48%', alignItems: 'center', elevation: 1 },
//   statVal: { fontSize: 20, fontWeight: '700', color: '#1F2937' },
//   statLabel: { fontSize: 12, color: '#6B7280', marginTop: 4 },
//   actionBtn: { backgroundColor: '#10B981', padding: 16, borderRadius: 12, alignItems: 'center', marginBottom: 20 },
//   disabled: { backgroundColor: '#A7F3D0', opacity: 0.6 },
//   actionBtnText: { color: '#FFF', fontSize: 18, fontWeight: '700' },
//   logTitle: { fontSize: 16, fontWeight: '600', color: '#374151', marginBottom: 8 },
//   logContainer: { backgroundColor: '#1E293B', padding: 12, borderRadius: 12, flex: 1 },
//   logText: { color: '#38BDF8', fontFamily: 'monospace', fontSize: 11, marginBottom: 4 }
// });


// import React, { useState, useRef, useEffect } from 'react';
// import { 
//   StyleSheet, 
//   Text, 
//   View, 
//   TouchableOpacity, 
//   ScrollView, 
//   ActivityIndicator, 
//   SafeAreaView 
// } from 'react-native';
// import { Asset } from 'expo-asset';
// // Ensure this matches your exact package name
// // import TorchModule from 'react-native-executorch'; 
// //  IMPORT VIA NATIVE MODULES WRAPPER
// import TorchModule from 'react-native-executorch';

// export default function App() {
//   const [logs, setLogs] = useState([]);
//   const [isTraining, setIsTraining] = useState(false);
//   const scrollViewRef = useRef();

//   // Helper function to append detailed logs with timestamps
//   const addLog = (message, type = 'INFO') => {
//     const timestamp = new Date().toLocaleTimeString();
//     const formattedLog = `[${timestamp}] [${type}] ${message}`;
//     setLogs((prevLogs) => [...prevLogs, formattedLog]);
//   };

//   // Auto-scroll to the bottom of the terminal when new logs stream in
//   useEffect(() => {
//     if (scrollViewRef.current) {
//       scrollViewRef.current.scrollToEnd({ animated: true });
//     }
//   }, [logs]);

//   const runOnDeviceProcessing = async () => {
//     if (isTraining) return;
//     setIsTraining(true);
//     setLogs([]); // Clear previous run console
    
//     addLog('Initialization started...', 'START');

//     try {
//       // 1. CRITICAL CRASH FIX: Unpack the binary .pte file from Expo assets to local storage
//       addLog('Resolving local ExecuTorch model binary asset...');
//       const modelModule = require('./assets/model.pte'); // Verify this path matches your file!
//       const [{ localUri }] = await Asset.loadAsync(modelModule);
      
//       if (!localUri) {
//         throw new Error('Failed to resolve local filesystem URI for the model.');
//       }
//       addLog(`Model cached successfully at: ${localUri.substring(0, 30)}...`, 'SUCCESS');

//       // 2. Load model into memory via C++ JSI Layer
//       addLog('Loading model into native ExecuTorch runtime engine...');
//       await TorchModule.loadModel(localUri);
//       addLog('Model successfully mapped into device memory context.', 'SUCCESS');

//       // 3. Simulated/Actual Training Loop with Verbose Telemetry
//       const localEpochs = 5;
//       addLog(`Starting local Edge Training loop (${localEpochs} epochs configured)...`, 'TRAIN');

//       for (let epoch = 1; epoch <= localEpochs; epoch++) {
//         addLog(`Epoch ${epoch}/${localEpochs} - Fetching local data batch...`, 'LOOP');
        
//         // Simulating the actual forward/backward processing step
//         // In your real setup, replace this line with your actual TorchModule execution:
//         // const result = await TorchModule.forward(inputTensorData);
//         await new Promise(resolve => setTimeout(resolve, 1200)); 

//         // Simulated loss reporting
//         const mockLoss = (0.45 / (epoch + 0.1)).toFixed(4);
//         const mockAccuracy = (72.5 + (epoch * 4.2)).toFixed(2);
        
//         addLog(`Epoch ${epoch} complete metrics -> Loss: ${mockLoss} | Accuracy: ${mockAccuracy}%`, 'METRIC');
//       }

//       // 4. Gradient Extraction & Serialization
//       addLog('Computing local gradients and delta adjustments...', 'PROCESSING');
//       await new Promise(resolve => setTimeout(resolve, 800));
//       addLog('Local model weights successfully serialized into payload buffer.', 'SUCCESS');
      
//       addLog('Ready for Federated Server synchronization pass! 📡', 'DONE');

//     } catch (error) {
//       // Catches and logs native C++ crashes cleanly instead of blowing up the app interface
//       addLog(`${error.message}`, 'CRITICAL_ERROR');
//       console.error(error);
//     } finally {
//       setIsTraining(false);
//     }
//   };

//   return (
//     <SafeAreaView style={styles.container}>
//       {/* Header Profile Info */}
//       <View style={styles.header}>
//         <Text style={styles.title}>Federated Edge Engine</Text>
//         <Text style={styles.subtitle}>Status: Connected to Tailscale Network</Text>
//       </View>

//       {/* FIXED: The Scrollable Terminal Log View Component */}
//       <View style={styles.terminalContainer}>
//         <View style={styles.terminalHeader}>
//           <Text style={styles.terminalHeaderText}>System Logging Output Console</Text>
//         </View>
        
//         <ScrollView 
//           style={styles.terminalScroll}
//           ref={scrollViewRef}
//           contentContainerStyle={{ paddingBottom: 20 }}
//         >
//           {logs.length === 0 ? (
//             <Text style={styles.placeholderText}>Awaiting device execution pipeline triggers...</Text>
//           ) : (
//             logs.map((log, index) => {
//               // Color code logs depending on severity level
//               let logColor = '#dcdccc';
//               if (log.includes('[CRITICAL_ERROR]')) logColor = '#ff6b6b';
//               if (log.includes('[SUCCESS]')) logColor = '#86af87';
//               if (log.includes('[METRIC]')) logColor = '#f0dfaf';
//               if (log.includes('[TRAIN]')) logColor = '#8cd0d3';

//               return (
//                 <Text key={index} style={[styles.logText, { color: logColor }]}>
//                   {log}
//                 </Text>
//               );
//             })
//           )}
//         </ScrollView>
//       </View>

//       {/* Action Button Controls */}
//       <TouchableOpacity 
//         style={[styles.button, isTraining && styles.buttonDisabled]} 
//         onPress={runOnDeviceProcessing}
//         disabled={isTraining}
//       >
//         {isTraining ? (
//           <View style={styles.loadingRow}>
//             <ActivityIndicator color="#fff" style={{ marginRight: 10 }} />
//             <Text style={styles.buttonText}>Processing Tensors...</Text>
//           </View>
//         ) : (
//           <Text style={styles.buttonText}>Execute On Device Processing</Text>
//         )}
//       </TouchableOpacity>
//     </SafeAreaView>
//   );
// }

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     backgroundColor: '#1c1c1c',
//     padding: 16,
//   },
//   header: {
//     marginBottom: 20,
//     marginTop: 10,
//   },
//   title: {
//     fontSize: 24,
//     fontWeight: 'bold',
//     color: '#ffffff',
//   },
//   subtitle: {
//     fontSize: 14,
//     color: '#9afbc5',
//     marginTop: 4,
//   },
//   terminalContainer: {
//     flex: 1,
//     backgroundColor: '#000000',
//     borderRadius: 8,
//     borderWidth: 1,
//     borderColor: '#3f3f3f',
//     overflow: 'hidden',
//     marginBottom: 20,
//   },
//   terminalHeader: {
//     backgroundColor: '#2b2b2b',
//     padding: 10,
//     borderBottomWidth: 1,
//     borderColor: '#3f3f3f',
//   },
//   terminalHeaderText: {
//     color: '#aaaaaa',
//     fontSize: 12,
//     fontFamily: 'monospace',
//     fontWeight: 'bold',
//   },
//   terminalScroll: {
//     flex: 1,
//     padding: 12,
//   },
//   logText: {
//     fontFamily: 'monospace',
//     fontSize: 13,
//     lineHeight: 18,
//     marginBottom: 6,
//   },
//   placeholderText: {
//     color: '#555555',
//     fontFamily: 'monospace',
//     fontSize: 13,
//     fontStyle: 'italic',
//   },
//   button: {
//     backgroundColor: '#388e3c',
//     paddingVertical: 16,
//     borderRadius: 8,
//     alignItems: 'center',
//     justifyContent: 'center',
//     elevation: 2,
//     marginBottom: 32,       // Pushes the button cleanly above the physical device bezel bar
//     marginHorizontal: 4,    // Keeps a balanced spacing profile on the screen edges
//   },
//   buttonDisabled: {
//     backgroundColor: '#2e5b32',
//   },
//   buttonText: {
//     color: '#ffffff',
//     fontSize: 16,
//     fontWeight: '600',
//   },
//   loadingRow: {
//     flexDirection: 'row',
//     alignItems: 'center',
//   }
// });




import React, { useState, useRef, useEffect } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TouchableOpacity, 
  ScrollView, 
  ActivityIndicator, 
  SafeAreaView,
  NativeModules // 👈 Loaded for custom bridging fallback
} from 'react-native';
import { Asset } from 'expo-asset';

// Safe module resolver extraction
const TorchModule = NativeModules.TorchModule || NativeModules.ExecutorchModule;

export default function OnDeviceTraining() {
  const [logs, setLogs] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const scrollViewRef = useRef();

  const addLog = (message, type = 'INFO') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prevLogs) => [...prevLogs, `[${timestamp}] [${type}] ${message}`]);
  };

  useEffect(() => {
    if (scrollViewRef.current) {
      scrollViewRef.current.scrollToEnd({ animated: true });
    }
  }, [logs]);

  const runOnDeviceProcessing = async () => {
    if (isTraining) return;
    setIsTraining(true);
    setLogs([]); 
    
    addLog('Initialization started...', 'START');

    // 💡 DIAGNOSTIC CODE: Dump all available native module names
    const availableModules = Object.keys(NativeModules);
    console.log("Available Modules:", availableModules);
    addLog(`Detected ${availableModules.length} native modules inside this APK structure.`, 'INFO');
    
    // Look for anything related to torch or executorch
    const matchingModules = availableModules.filter(m => 
      m.toLowerCase().includes('torch') || m.toLowerCase().includes('exec')
    );
    
    if (matchingModules.length > 0) {
      addLog(`Found potential matching modules: ${matchingModules.join(', ')}`, 'SUCCESS');
    } else {
      addLog('WARNING: No native module containing "torch" or "exec" found in this APK binary binary wrapper.', 'LOOP');
    }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Federated Edge Engine</Text>
        <Text style={styles.subtitle}>Status: Connected via Tailscale Tunnel</Text>
      </View>

      <View style={styles.terminalContainer}>
        <View style={styles.terminalHeader}>
          <Text style={styles.terminalHeaderText}>System Logging Output Console</Text>
        </View>
        
        <ScrollView 
          style={styles.terminalScroll}
          ref={scrollViewRef}
          contentContainerStyle={{ paddingBottom: 20 }}
        >
          {logs.length === 0 ? (
            <Text style={styles.placeholderText}>Awaiting device execution pipeline triggers...</Text>
          ) : (
            logs.map((log, index) => {
              let logColor = '#dcdccc';
              if (log.includes('[CRITICAL_ERROR]')) logColor = '#ff6b6b';
              if (log.includes('[SUCCESS]')) logColor = '#86af87';
              if (log.includes('[METRIC]')) logColor = '#f0dfaf';
              return <Text key={index} style={[styles.logText, { color: logColor }]}>{log}</Text>;
            })
          )}
        </ScrollView>
      </View>

      <TouchableOpacity 
        style={[styles.button, isTraining && styles.buttonDisabled]} 
        onPress={runOnDeviceProcessing}
        disabled={isTraining}
      >
        {isTraining ? (
          <View style={styles.loadingRow}>
            <ActivityIndicator color="#fff" style={{ marginRight: 10 }} />
            <Text style={styles.buttonText}>Processing Tensors...</Text>
          </View>
        ) : (
          <Text style={styles.buttonText}>Execute On Device Processing</Text>
        )}
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1c1c1c',
    padding: 16,
  },
  header: {
    marginBottom: 20,
    marginTop: 10,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
  },
  subtitle: {
    fontSize: 14,
    color: '#9afbc5',
    marginTop: 4,
  },
  terminalContainer: {
    flex: 1,
    backgroundColor: '#000000',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#3f3f3f',
    overflow: 'hidden',
    marginBottom: 20,
  },
  terminalHeader: {
    backgroundColor: '#2b2b2b',
    padding: 10,
    borderBottomWidth: 1,
    borderColor: '#3f3f3f',
  },
  terminalHeaderText: {
    color: '#aaaaaa',
    fontSize: 12,
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  terminalScroll: {
    flex: 1,
    padding: 12,
  },
  logText: {
    fontFamily: 'monospace',
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 6,
  },
  placeholderText: {
    color: '#555555',
    fontFamily: 'monospace',
    fontSize: 13,
    fontStyle: 'italic',
  },
  button: {
    backgroundColor: '#388e3c',
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 2,
    marginBottom: 32,       // 👈 Pushes button cleanly above the system gesture lines
    marginHorizontal: 4,
  },
  buttonDisabled: {
    backgroundColor: '#2e5b32',
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
  }
});
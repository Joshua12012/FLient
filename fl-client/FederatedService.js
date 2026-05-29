import { Alert } from 'react-native';
import { initExecutorch, CustomModelModule } from 'react-native-executorch';
import { ExpoResourceFetcher } from 'react-native-executorch-expo-resource-fetcher'; // Change to BareResourceFetcher if not using Expo
import * as FileSystem from 'expo-file-system';

// Initialize the background native bridge wrapper layer
initExecutorch({ resourceFetcher: ExpoResourceFetcher });

// Point to your laptop's Wi-Fi network IP
const LAPTOP_IP = "100.87.17.13"; // 👈 REPLACE THIS with your actual laptop IP
const SERVER_DOWNLOAD_URL = `http://${LAPTOP_IP}:8001/download_model`;
const SERVER_UPLOAD_URL = `http://${LAPTOP_IP}:8001/upload_weights`;

const LOCAL_MODEL_PATH = `${FileSystem.documentDirectory}downloaded_global_model.pte`;

export const runMobileFederatedRound = async () => {
  try {
    console.log("📡 Pinging central coordinator server...");
    
    // 1. Over-the-Air Model Download
    const downloadRes = await FileSystem.downloadAsync(
      SERVER_DOWNLOAD_URL,
      LOCAL_MODEL_PATH
    );
    
    if (downloadRes.status !== 200) {
      throw new Error(`Server deployment error code: ${downloadRes.status}`);
    }
    console.log("📥 New .pte binary successfully pulled and cached on storage!");

    // 2. Load the custom model graph straight into the ExecuTorch runtime engine
    console.log("⚙️ Loading model graph into mobile hardware layout...");
    const modelInstance = await CustomModelModule.fromCustomModel({
      modelSource: LOCAL_MODEL_PATH,
    });

    // 3. Generate mock input features matching your 10-feature framework structure
    const mockSensorInputs = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    
    console.log("🚀 Executing native on-device forward inference pass...");
    const predictionOutputs = await modelInstance.forward([mockSensorInputs]);
    console.log("🎉 Inference completed natively on phone! Outputs:", predictionOutputs);

    // 4. Ship simulated training updates back to the aggregator pool
    console.log("📤 Serializing gradients delta payload...");
    const formData = new FormData();
    
    // Attach the model binary stream to the multi-part form
    formData.append('file', {
      uri: LOCAL_MODEL_PATH,
      name: 'mobile_weights.pt',
      type: 'application/octet-stream',
    });

    const uploadRes = await fetch(SERVER_UPLOAD_URL, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    const uploadJson = await uploadRes.json();
    console.log("📡 Aggregator Response:", uploadJson);
    
    Alert.alert("Success", "Federated check-in complete on mobile node!");
    
  } catch (error) {
    console.error("❌ Federated round failed:", error);
    Alert.alert("Execution Error", error.message);
  }
};
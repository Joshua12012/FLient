const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Force Metro to bundle the ExecuTorch .pte files correctly 
config.resolver.assetExts.push('pte');

module.exports = config;
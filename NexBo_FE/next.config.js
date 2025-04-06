/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    domains: ['localhost', 'firebasestorage.googleapis.com'],
  },
  webpack: (config, { isServer }) => {
    // Add a fallback for the @splinetool/react-spline package
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        module: false,
      };
    }
    
    return config;
  },
  // Prevent build errors related to imported modules
  transpilePackages: ['@splinetool/react-spline', '@splinetool/runtime'],
  async rewrites() {
    return [
      {
        source: '/chat',
        destination: 'http://127.0.0.1:8000/chat',
      },
    ];
  },
  typescript: {
    ignoreBuildErrors: true
  }
};

module.exports = nextConfig;
/** @type {import('next').NextConfig} */
const nextConfig = {
    async rewrites() {
        return [
            {
                source: '/chat',
                destination: 'http://127.0.0.1:8000/chat',
            },
        ];
    },
};

module.exports = nextConfig;
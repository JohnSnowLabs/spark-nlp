'use strict';

const { resolve } = require('path');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';
  return {
    mode: isProduction ? 'production' : 'development',
    entry: { models: './src/models/index.js' },
    output: {
      filename: '[name].js',
      publicPath: '/static/',
      path: resolve(__dirname, 'static'),
    },
    module: {
      rules: [
        {
          test: /\.js$/,
          exclude: /(node_modules)/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: ['@babel/preset-env', '@babel/preset-react'],
            },
          },
        },
        {
          test: /\.css$/,
          use: [
            'style-loader',
            {
              loader: 'css-loader',
            },
          ],
        },
      ],
    },
    devServer: {
      port: 9000,
      hot: true,
      proxy: {
        '/': 'http://localhost:4000',
      },
    },
  };
};

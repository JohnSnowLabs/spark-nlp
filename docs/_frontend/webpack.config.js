'use strict';

const { resolve } = require('path');
const webpack = require('webpack');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const TerserPlugin = require('terser-webpack-plugin');
const CssMinimizerPlugin = require('css-minimizer-webpack-plugin');

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
          exclude: /\.module\.css$/,
          use: [
            isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
            {
              loader: 'css-loader',
              options: {
                importLoaders: 1,
              },
            },
            {
              loader: 'postcss-loader',
              options: {
                postcssOptions: {
                  plugins: [
                    [
                      'postcss-preset-env',
                      {
                        stage: 2,
                        features: {
                          'nesting-rules': true,
                        },
                      },
                    ],
                  ],
                },
              },
            },
          ],
        },
        {
          test: /\.module\.css$/,
          use: [
            isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
            {
              loader: 'css-loader',
              options: {
                modules: true,
                importLoaders: 1,
              },
            },
            {
              loader: 'postcss-loader',
              options: {
                postcssOptions: {
                  plugins: [
                    [
                      'postcss-preset-env',
                      {
                        stage: 2,
                        features: {
                          'nesting-rules': true,
                        },
                      },
                    ],
                  ],
                },
              },
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
    optimization: {
      moduleIds: 'deterministic',
      minimize: isProduction,
      minimizer: isProduction
        ? [new TerserPlugin(), new CssMinimizerPlugin()]
        : undefined,
    },
    plugins: [
      new webpack.EnvironmentPlugin({
        SEARCH_ORIGIN:
          process.env.SEARCH_ORIGIN ||
          'https://search.modelshub.johnsnowlabs.com',
      }),
      isProduction &&
        new MiniCssExtractPlugin({
          filename: '[name].css',
        }),
    ].filter(Boolean),
  };
};

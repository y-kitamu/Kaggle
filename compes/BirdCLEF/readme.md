# BirdCLEF

## Description

## Feature

フレームワーク:
- tfx
- hydra (omegaconf)

## 実験管理
パラメータ管理はyaml形式で、読み込み等にはhydraを使用する。(omegaconfだけでも事足りる?)
ただし、ソースコード内のtypingにはdataclasssを使用する。

#!/bin/bash

cd input
kaggle datasets download -d yujiariyasu/rsna2024-oof
unzip rsna2024-oof.zip
cd ..
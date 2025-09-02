#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from modelo_unet.main import main

if __name__ == "__main__":
    print("Iniciando treinamento do modelo U-Net...")
    main()
    print("Treinamento concluido!")
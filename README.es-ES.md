<div align=center>

# Borrador, Verificar, Restaurar: Restauración de Inscripciones Históricas con AutoRefinamiento mediante un MLLM Unificado

</div>

<div align=center>

[![Paper](https://img.shields.io/badge/Paper-ACL2026-ff6b6b)](https://aclanthology.org/2026.acl-long.1254/)
[![GitHub ZZXF11](https://img.shields.io/badge/GitHub-ZZXF11-blueviolet?logo=github)](https://github.com/ZZXF11)
[![SCUT DLVC Lab](https://img.shields.io/badge/SCUT-DLVC_Lab-327FE6?logo=Academia&logoColor=white)](http://dlvc-lab.net/lianwen/)
[![Code](https://img.shields.io/badge/Code-UniHIR-yellow)](https://github.com/ZZXF11/UniHIR)

</div>

## Importante
Los datos originales del conjunto de datos provienen de canales públicos como Internet, y su derecho de autor permanecerá con los proveedores originales. El conjunto de datos recopilado y anotado presentado en este caso es solo para uso no comercial y actualmente está licenciado a universidades e instituciones de investigación. Para solicitar el uso de este conjunto de datos, complete el formulario de solicitud correspondiente de acuerdo con los requisitos especificados en el sitio web oficial del conjunto de datos. El solicitante debe ser un empleado a tiempo completo de una universidad o instituto de investigación y se requiere firmar el formulario de solicitud. Para facilitar la revisión, se recomienda adjuntar un sello oficial (un sello de un departamento de nivel secundario es aceptable).

## 🌟 Puntos Destacados
- **UniHIR**
![Vis_1](fig/unihir_pipeline.png)
- **HIRBench**
![Vis_2](fig/hirbench.png)

- Proponemos **UniHIR**, un pionero MLLM Unificado para Restauración de Inscripciones Históricas (HIR) de extremo a extremo, ofreciendo una nueva perspectiva sobre HIR.
- UniHIR incorpora dos diseños novedosos, Localización Guiada por Borrador y AutoRefinamiento Jerárquico, para soportar razonamiento iterativo y auto-corrección para HIR.
- Proponemos UHIRFactory y construimos **HIRBench** para permitir entrenamiento paso a paso y eficiente en memoria de UniHIR.
- Experimentos extensivos demuestran que nuestro método logra una precisión superior del texto restaurado y calidad de generación.

## 📏 Resultado de Evaluación
![Vis_3](fig/eval.png)

## 🌄 Visualización
![vis_4](fig/vis1.png)

## 📅 Noticias
- **2026.07.05**: 🎉🎉 Nuestro [paper](https://aclanthology.org/2026.acl-long.1254/) es aceptado por ACL Main.


## 🚧 Lista de Tareas Pendientes

- [x] Publicar código de inferencia
- [x] Publicar modelo UniHIR
- [ ] Publicar HIRBench

## 🚧 Instalación
Clonar este repositorio:
```bash
git clone https://github.com/ZZXF11/UniHIR
cd UniHIR
```

**Paso 0**: Descargar e instalar Miniconda desde el [sitio web oficial](https://docs.conda.io/en/latest/miniconda.html).

**Paso 1**: Crear un entorno conda y activarlo.
```bash
conda create -n unihir python=3.10 -y
conda activate unihir
```

**Paso 2**: Instalar los paquetes necesarios.
```bash
pip install -r requirements.txt
```

## 📺 Inferencia

**Paso 1**: Descargar el modelo preentrenado desde [ModelScope](https://www.modelscope.cn/models/zzxfzyy/UniHIR) o [Hugging Face](https://www.modelscope.cn/models/zzxfzyy/UniHIR).

**Paso 2**: Ejecutar inferencia:
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python infer.py \
    --img_path ejemplos/FS_12_159_2.jpg \
    --ref_count 6 \
    --save_path ./resultados \
    --model_path ./UniHIR
```

> **Nota**: Recomendamos usar una GPU con 80GB de VRAM (por ejemplo, NVIDIA A100) para inferencia.

### Argumentos

| Argumento | Tipo | Predeterminado | Descripción |
|----------|------|---------|-------------|
| `--img_path` | str | `ejemplos/FS_12_159_2.jpg` | Ruta de la imagen del documento histórico dañado de entrada |
| `--ref_count` | int | `6` | Número de iteraciones de refinamiento |
| `--save_path` | str | `./resultados` | Directorio para guardar los resultados de inferencia |
| `--model_path` | str | `./UniHIR` | Ruta del directorio del modelo UniHIR preentrenado |



## 🔥 HIRBench
| Conjunto de Datos | Descripción | Enlace | Estado |
|---------|-------------|------|--------|
| HIRBench-DGL | Localización Guiada por Borrador | [Descargar](#) | 🚧 Próximamente |
| HIRBench-HSR | AutoRefinamiento Jerárquico | [Descargar](#) | 🚧 Próximamente |
| HIRBench-AR | Restauración de Apariencia | [Descargar](#) | 🚧 Próximamente |


**Nota:**
- HIRBench solo puede usarse para propósitos de investigación no comercial. Los académicos u organizaciones que deseen utilizar HIRBench pueden aplicar a través de nuestra plataforma en línea: 👉 [Aplicar Aquí](http://121.41.49.212:9000/)
- Le proporcionaremos la contraseña de descompresión después de recibir y aprobar su aplicación.
- Todos los usuarios deben seguir todas las condiciones de uso; de lo contrario, la autorización será revocada.

## 💙 Agradecimiento
- [BAGEL](https://github.com/bytedance-seed/BAGEL)
- [AutoHDR](https://github.com/SCUT-DLVCLab/AutoHDR)
- [DiffHDR](https://github.com/yeungchenwa/HDR)
- [HisDoc1B](https://github.com/SCUT-DLVCLab/HisDoc1B)
- [MegaHan97K](https://github.com/SCUT-DLVCLab/MegaHan97K)


## 📜 Licencia
El código y el conjunto de datos deben usarse y distribuirse bajo [(CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/) para fines de investigación no comercial.

## ⛔️ Derechos de Autor
- Este repositorio solo puede usarse para fines de investigación no comercial.
- Para uso comercial, por favor póngase en contacto con el Profesor Lianwen Jin (eelwjin@scut.edu.cn).
- Derechos de autor 2026, [Laboratorio de Aprendizaje Profundo y Procesamiento de Imágenes (DLVC-Lab)](http://www.dlvc-lab.net), Universidad del Sur de China de Tecnología.

## ✒️Cita
Si encuentra que UniHIR es útil, considere darle ⭐ a este repositorio y citar:
```latex
@article{zhang2026unihir,
      title={Borrador, Verificar, Restaurar: Restauración de Inscripciones Históricas con AutoRefinamiento mediante un MLLM Unificado}, 
      author={Yuyi Zhang, Junle Liu, Peirong Zhang, Jianliang Liu, Zhenhua Yang, Lianwen Jin},
      journal={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
      year={2026},
}
```
¡Gracias por su apoyo!

## ⭐ Historial de Estrellas
[![Star Rising](https://api.star-history.com/svg?repos=ZZXF11/UniHIR&type=Timeline)](https://star-history.com/#ZZXF11/UniHIR&Timeline)

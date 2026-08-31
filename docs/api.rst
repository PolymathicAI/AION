API Reference
=============

This page provides comprehensive API documentation for all AION components, automatically generated from the source code.

.. currentmodule:: aion

Main Model
----------

.. automodule:: aion.model
   :members:
   :undoc-members:
   :show-inheritance:

Modalities
----------

The modality system defines data structures for all 39 astronomical data types supported by AION.

Base Classes
~~~~~~~~~~~~

.. automodule:: aion.modalities
   :members: Modality, Image, Spectrum, Scalar
   :undoc-members:
   :show-inheritance:

Image Modalities
~~~~~~~~~~~~~~~~

.. autoclass:: aion.modalities.LegacySurveyImage
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCImage
   :members:
   :undoc-members:
   :show-inheritance:

Spectrum Modalities
~~~~~~~~~~~~~~~~~~~

.. autoclass:: aion.modalities.DESISpectrum
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.SDSSSpectrum
   :members:
   :undoc-members:
   :show-inheritance:

Catalog Modalities
~~~~~~~~~~~~~~~~~~

.. autoclass:: aion.modalities.LegacySurveyCatalog
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveySegmentationMap
   :members:
   :undoc-members:
   :show-inheritance:

Scalar Modalities
~~~~~~~~~~~~~~~~~

Legacy Survey Scalars
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: aion.modalities.LegacySurveyFluxG
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyFluxR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyFluxI
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyFluxZ
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyFluxW1
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyFluxW2
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyFluxW3
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyFluxW4
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyShapeR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyShapeE1
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyShapeE2
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.LegacySurveyEBV
   :members:
   :undoc-members:
   :show-inheritance:

HSC Scalars
~~~~~~~~~~~

.. autoclass:: aion.modalities.HSCAG
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCAR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCAI
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCAZ
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCAY
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCMagG
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCMagR
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCMagI
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCMagZ
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCMagY
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCShape11
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCShape22
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.HSCShape12
   :members:
   :undoc-members:
   :show-inheritance:

Gaia Scalars
~~~~~~~~~~~~

.. autoclass:: aion.modalities.GaiaFluxG
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.GaiaFluxBp
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.GaiaFluxRp
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.GaiaParallax
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.GaiaXpBp
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.GaiaXpRp
   :members:
   :undoc-members:
   :show-inheritance:

Coordinate Scalars
~~~~~~~~~~~~~~~~~~

.. autoclass:: aion.modalities.Ra
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.Dec
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: aion.modalities.Z
   :members:
   :undoc-members:
   :show-inheritance:

Utility Types
~~~~~~~~~~~~~

.. py:data:: ScalarModalities

   Mapping from scalar modality names to their corresponding modality classes.

.. py:data:: ModalityType

   Union type covering all supported modality data structures.

Codec System
------------

The codec system handles tokenization of different modality types.

Core Codec Classes
~~~~~~~~~~~~~~~~~~

.. automodule:: aion.codecs.manager
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.base
   :members:
   :undoc-members:
   :show-inheritance:

Codec Implementations
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.codecs.image
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.spectrum
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.catalog
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.scalar_field
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.scalar
   :members:
   :undoc-members:
   :show-inheritance:

Quantizers
~~~~~~~~~~

.. automodule:: aion.codecs.quantizers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.quantizers.scalar
   :members:
   :undoc-members:
   :show-inheritance:

4M Transformer
--------------

Core transformer architecture and components.

Main Transformer
~~~~~~~~~~~~~~~~

.. automodule:: aion.fourm.fm
   :members:
   :undoc-members:
   :show-inheritance:

Embedding Layers
~~~~~~~~~~~~~~~~

.. automodule:: aion.fourm.encoder_embeddings
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.fourm.decoder_embeddings
   :members:
   :undoc-members:
   :show-inheritance:

Transformer Components
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.fourm.fm_utils
   :members:
   :undoc-members:
   :show-inheritance:

Generation
~~~~~~~~~~

.. automodule:: aion.fourm.generate
   :members:
   :undoc-members:
   :show-inheritance:

LoRA Support
~~~~~~~~~~~~

.. automodule:: aion.fourm.lora_utils
   :members:
   :undoc-members:
   :show-inheritance:

Modality Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.fourm.modality_info
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.fourm.modality_transforms
   :members:
   :undoc-members:
   :show-inheritance:

Codec Modules
-------------

Specialized neural network modules used in codecs.

Architecture Components
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.codecs.modules.magvit
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.modules.convnext
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.modules.convblocks
   :members:
   :undoc-members:
   :show-inheritance:

Specialized Modules
~~~~~~~~~~~~~~~~~~~

.. automodule:: aion.codecs.modules.spectrum
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.modules.ema
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: aion.codecs.modules.subsampler
   :members:
   :undoc-members:
   :show-inheritance:

Configuration and Utilities
----------------------------

.. automodule:: aion.codecs.config
   :members:
   :undoc-members:
   :show-inheritance:

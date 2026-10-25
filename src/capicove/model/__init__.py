"""Verified local assets, offline profiles and bounded typed model factories."""

from capicove.model.bundle import (
    BundleArtifact,
    ImageEncoderBundleManifest,
    ImagePreprocessingContract,
    OnnxOutputContract,
    OnnxRuntimeContract,
    OnnxTensorContract,
    VerifiedImageEncoderBundle,
    load_verified_image_encoder_bundle,
)
from capicove.model.configuration import (
    ModelCapability,
    ModelConfiguration,
    ModelProfile,
    load_configuration,
)
from capicove.model.factory import ModelCapacityError, ModelFactory
from capicove.model.http_profile import HttpModelProfile
from capicove.model.manifest import (
    ImageModelManifest,
    ImageModelSpec,
    load_manifest,
    load_packaged_manifest,
)
from capicove.model.receipt import (
    VerifiedModelInstallation,
    verify_model_installation,
)
from capicove.model.registry import ProviderDefinition, ProviderRegistry


__all__ = [
    "BundleArtifact",
    "HttpModelProfile",
    "ImageEncoderBundleManifest",
    "ImageModelManifest",
    "ImageModelSpec",
    "ImagePreprocessingContract",
    "ModelCapability",
    "ModelCapacityError",
    "ModelConfiguration",
    "ModelFactory",
    "ModelProfile",
    "OnnxOutputContract",
    "OnnxRuntimeContract",
    "OnnxTensorContract",
    "ProviderDefinition",
    "ProviderRegistry",
    "VerifiedImageEncoderBundle",
    "VerifiedModelInstallation",
    "load_configuration",
    "load_manifest",
    "load_packaged_manifest",
    "load_verified_image_encoder_bundle",
    "verify_model_installation",
]

"""
Renaissance Text GAN

A GAN-based model for generating synthetic Renaissance-style text with 
realistic printing imperfections like ink bleed, smudging, and faded text.
"""

from renaissance_text_gan.models import (
    Generator, 
    Discriminator, 
    DegradationLayer,
    RenaissanceTextGenerator
)

__version__ = '0.1.0'
__author__ = 'Renaissance Text GAN Team'

__all__ = [
    'Generator',
    'Discriminator',
    'DegradationLayer',
    'RenaissanceTextGenerator'
] 
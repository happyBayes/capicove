# Image OCR

`ImageOcr` converts an authorized local image into structured OCR words and image
evidence. The Tesseract adapter uses an installed local executable and configured
language data; it does not send images to a hosted service.

The default language set is exposed as `DEFAULT_OCR_LANGUAGES`. Verify that every
requested Tesseract language pack is installed before invocation. OCR output
includes word text, confidence, bounding boxes, source identity, and adapter
lineage.

See the [image API](../../api/image.md) for the public operation and result types.

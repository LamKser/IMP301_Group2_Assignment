from glob import glob


from panorama.stitcher import Stitcher


stitcher = Stitcher()
panorama = stitcher.stitch(glob('test_panorama/test_1/*.jpg'))

# settings = {"detector": "sift", "confidence_threshold": 0.2}
# stitcher = Stitcher(**settings)
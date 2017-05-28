import lanefinder
import thresholding
import utils

class Pipeline:

	def __init__(self, mtx, coeffs, M, Minv):
		self.detector = lanefinder.LaneDetector()
		self.lines = []
		self.mtx = mtx
		self.coeffs = coeffs
		self.M = M
		self.Minv = Minv

	def process_frame(self, frame):
		binary_warped = thresholding.threshold(
			frame,
			self.mtx,
			self.coeffs,
			self.M
		)
		self.detector.detect(binary_warped)
		lane = self.detector.draw_lane(binary_warped, self.Minv)
		frame = utils.weighted_img(frame, lane)
		return frame

	def __call__(self, frame):
		return self.process_frame(frame)


if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt

    camdata = pickle.load(open('./camera_cal/camera_cal_pickle.p', 'rb'))
    mtx = camdata['mtx']
    dist = camdata['dist']
    persp = pickle.load(open('./camera_cal/persp_pickle.p', 'rb'))
    M = persp['M']
    Minv = persp['Minv']

    test_img = plt.imread('./test_images/test1.jpg')
    p = Pipeline(mtx, dist, M, Minv)
    res = p.process_frame(test_img)
    plt.imsave('./output_images/frame.png', res, cmap='gray')
    

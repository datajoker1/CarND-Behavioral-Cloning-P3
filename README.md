# P3: Behavioural Cloning
 This project uses deep neural networks and convolutional neural networks to clone driving behavior. Using a set of images and predicted steering angles, a CNN is generates with Keras to predict steering angles from images using an Udaciy simulator.

##  Included Files
* `model.py`: file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
* `model.h5`: containing the trained convolution neural network
* `drive.py`: for driving the car in autonomous mode

## 2. Dataset Characteristics

### Data Generation: Udacity's Car-Driving Simulator
I made the train data myself.
I tried to act like a good driver, At last I managed to keep the track in the middle.


## 3. Solution Design
As the goal was to predict steering angles on a road; I cropped out unuseful portions of the images to reduce what the network had to learn. After augumenting the data set; I then used Convolutional networks,Pooling layers and Fully connected Layers for my model; I used mean squared error has my loss function since this is a regression problem,adam optimizer; I used Dropout as a regularization Technique. The model is described in full below

## 4. Model architecture


<table>
	<th>Layer</th><th>Details</th>
	<tr>
		<td>Convolution Layer 1</td>
		<td>
			<ul>
				<li>Filters: 24</li>
				<li>Kernel: 5 x 5</li>
				<li>Stride: 2 x 2</li>
				<li>Padding: SAME</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 2</td>
		<td>
			<ul>
				<li>Filters: 36</li>
				<li>Kernel: 5 x 5</li>
				<li>Stride: 2 x 2</li>
				<li>Padding: SAME</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 3</td>
		<td>
			<ul>
				<li>Filters: 48</li>
				<li>Kernel: 5 x 5</li>
				<li>Stride: 2 x 2</li>
				<li>Padding: SAME</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 4</td>
		<td>
			<ul>
				<li>Filters: 64</li>
				<li>Kernel: 3 x 3</li>
				<li>Padding: SAME</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Convolution Layer 5</td>
		<td>
			<ul>
				<li>Filters: 64</li>
				<li>Kernel: 3 x 3</li>
				<li>Padding: SAME</li>
				<li>Activation: relu</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Flatten layer</td>
		<td>
			<ul>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 1</td>
		<td>
			<ul>
				<li>Neurons: 100</li>
				<li>DropOut: 0.5</li>
			</ul>
		</td>
	</tr>
   	<tr>
		<td>Fully Connected Layer 2</td>
		<td>
			<ul>
				<li>Neurons: 50</li>
				<li>DropOut: 0.5</li>
			</ul>
		</td>
	</tr>
	<tr>
		<td>Fully Connected Layer 3</td>
		<td>
			<ul>
				<li>Neurons: 1</li>
			</ul>
		</td>
	</tr>
</table>

## 5. Discussion

Adding the dropout layers was key in getting the model to work; while the validation losses were slightly lower without them.

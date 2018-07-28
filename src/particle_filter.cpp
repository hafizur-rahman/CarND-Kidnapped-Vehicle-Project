/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;

	// Resize weights vector based on num_particles
  	weights.resize(num_particles);

	// define normal distributions for sensor noise
	normal_distribution<double> noise_x(0, std[0]);
	normal_distribution<double> noise_y(0, std[1]);
	normal_distribution<double> noise_theta(0, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle p;

		p.id = i;
		p.x = x + noise_x(gen);
		p.y = y + noise_y(gen);
		p.theta = theta + noise_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// define normal distributions for sensor noise
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle& p = particles[i];

		// Calculate new state
		if (fabs(yaw_rate) < 0.00001) {
			p.x += velocity * cos(p.theta) * delta_t;
			p.y += velocity * sin(p.theta) * delta_t;
		} else {
			double theta_f = p.theta + yaw_rate*delta_t;
			p.x += velocity / yaw_rate * (sin(theta_f) - sin(p.theta));
			p.y += velocity / yaw_rate * (cos(p.theta) - cos(theta_f));
			p.theta = theta_f;
		}

		// Add noise
		p.x += noise_x(gen);
		p.y += noise_y(gen);
		p.theta += noise_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (uint i = 0; i < observations.size(); i ++) {
		LandmarkObs o = observations[i];

		double min_dist = numeric_limits<double>::max();
		int landmark_id = -1;

		for (uint j = 0; j < predicted.size(); j ++ ) {
			LandmarkObs p = predicted[j];

			// Calculate distance
			double distance = dist(o.x, o.y, p.x, p.y);
			if (distance < min_dist) {
				min_dist = distance;
				landmark_id = p.id;
			}
		}

		// Set the observation's id to the nearest predicted landmark's id
		observations[i].id = landmark_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// First term of multi-variate normal Gaussian distribution calculated below
  	// It stays the same so can be outside the loop
  	const double a = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  
	// The denominators of the mvGd also stay the same
	const double x_denom = 2 * std_landmark[0] * std_landmark[0];
	const double y_denom = 2 * std_landmark[1] * std_landmark[1];

	// Iterate through each particle
	for (int i = 0; i < num_particles; ++i) {
		Particle& p = particles[i];

		// Create a vector to hold the map landmark locations predicted to be within sensor range of the particle
    	vector<LandmarkObs> predictions;

		for (uint j = 0; j < map_landmarks.landmark_list.size(); j++) {
			const Map::single_landmark_s& lm = map_landmarks.landmark_list[j];
			
			if (dist(p.x, p.y, lm.x_f, lm.y_f) <= sensor_range) {
				predictions.push_back(LandmarkObs{ lm.id_i, lm.x_f, lm.y_f });
			}
		}

		// Transform the observation point (from vehicle coordinates to map coordinates)
		vector<LandmarkObs> transformed_os;
		for (unsigned int j = 0; j < observations.size(); j++) {
			const LandmarkObs& o = observations[j];

			double t_x = o.x * cos(p.theta) - o.y * sin(p.theta) + p.x;
			double t_y = o.x * sin(p.theta) + o.y * cos(p.theta) + p.y;
		
			transformed_os.push_back(LandmarkObs{ observations[j].id, t_x, t_y });
		}

		dataAssociation(predictions, transformed_os);

		// For calculating multi-variate Gaussian distribution of each observation, for each particle
		double mvGd = 1.0;

		for (unsigned int j = 0; j < transformed_os.size(); j++) {
			const LandmarkObs& o = transformed_os[j];
			
			double pr_x, pr_y;

			// Get the x,y coordinates of the prediction associated with the current observation
			for (uint k = 0; k < predictions.size(); k++) {
				if (predictions[k].id == o.id) {
					pr_x = predictions[k].x;
					pr_y = predictions[k].y;

					break;
				}
			}

			// Calculate multi-variate Gaussian distribution
			double b = (pow(o.x - pr_x, 2) / x_denom) + (pow(o.y - pr_y, 2) / y_denom);

			mvGd *= a * exp(-b);
		}	

		// Update particle weights with combined multi-variate Gaussian distribution
		particles[i].weight = mvGd;

		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	vector<Particle> new_particles;

	// generate random starting index for resampling wheel
	uniform_int_distribution<int> uniintdist(0, num_particles-1);
	auto index = uniintdist(gen);

	// Get max weight
	double max_weight = *max_element(weights.begin(), weights.end());

	// Uniform random distribution [0.0, max_weight)
	uniform_real_distribution<double> unirealdist(0.0, max_weight);

	double beta = 0.0;

	// Spin the resample wheel!
	for (int i = 0; i < num_particles; i++) {
		beta += unirealdist(gen) * 2.0;
		while (beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

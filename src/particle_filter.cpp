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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 100;

        default_random_engine gen;
        normal_distribution<double> dist_x(x, std[0]);
        normal_distribution<double> dist_y(y, std[1]);
        normal_distribution<double> dist_theta(std[2]);
        Particle p;

        for (int i = 0; i < num_particles; i++) {
           p.x = dist_x(gen);
           p.y = dist_y(gen);
           p.theta = dist_theta(gen);
           p.weight = 1.0; 
           particles.push_back(p);
           weights.push_back(1.0);
        }

        is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

        default_random_engine gen;
        normal_distribution<double> dist_x(0, std_pos[0]);
        normal_distribution<double> dist_y(0, std_pos[1]);
        normal_distribution<double> dist_theta(0, std_pos[2]);

        for (int i = 0; i < num_particles; i++) {
            double theta = particles[i].theta;
            if (yaw_rate > 0.001) {
                particles[i].x += velocity / yaw_rate * (sin(theta + delta_t * yaw_rate) - sin(theta));
                particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + delta_t * yaw_rate));
            }
            else {
                particles[i].x += velocity * delta_t * cos(theta);
                particles[i].y += velocity * delta_t * sin(theta);
            }
            particles[i].theta += yaw_rate * delta_t;

            // add noise
            particles[i].x += dist_x(gen);
            particles[i].y += dist_y(gen);
            particles[i].theta += dist_theta(gen);
        }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> &predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
        
        for (int i = 0; i < observations.size(); i++) {
            double minr = 99999999.0, cr;
            int min_id = -1;
            for (int j = 0; j < predicted.size(); j++) {
                cr = distSqr(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
                if (cr < minr) {
                    minr = cr;
                    min_id = predicted[j].id;
                }
            }
            observations[i].id = min_id;
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
        
        double stdx_c = std_landmark[0] * std_landmark[0] * 2;
        double stdy_c = std_landmark[1] * std_landmark[1] * 2;
        double rangeSqr = sensor_range * sensor_range;

        for (int i = 0; i < num_particles; i++) {
            double px = particles[i].x;
            double py = particles[i].y;
            double pt = particles[i].theta;
            double mx, my, ox, oy;
            int id;
            vector<LandmarkObs> m_obs;
            vector<LandmarkObs> marks;
            LandmarkObs ob;
            vector<double> sense_x;
            vector<double> sense_y;
            vector<int> associations;

            // coordinate transformation
            for (int j = 0; j < observations.size(); j++) {
                ox = observations[j].x;
                oy = observations[j].y;
                ob.x = px + cos(pt) * ox - sin(pt) * oy;
                ob.y = py + sin(pt) * ox + cos(pt) * oy;
                m_obs.push_back(ob);
                
                sense_x.push_back(ob.x);
                sense_y.push_back(ob.y);
            }
            // find landmarks within particles sensor range
            // questions here if only search the map within sensor range:
            // what if there's no landmarks within this particle's sensor range?
            // what if an observation is out of this particles's sensor range?
            // what if an observation is closer to a landmark that's out of this particle's sensor range?
            for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
                mx = map_landmarks.landmark_list[j].x_f;
                my = map_landmarks.landmark_list[j].y_f;
                id = map_landmarks.landmark_list[j].id_i;
                if (distSqr(px, py, mx, my) < rangeSqr) {
                    ob.x = mx;
                    ob.y = my;
                    ob.id = id;
                    marks.push_back(ob);
                }
	    }

            // association
            dataAssociation(marks, m_obs);
            
            for (int j = 0; j < m_obs.size(); j++)
                associations.push_back(m_obs[j].id);
            SetAssociations(particles[i], associations, sense_x, sense_y);
            
            // update weight
            double weight = 1.0;
            double assoc_x, assoc_y;
            //double gauss_norm;
            double exponent;
            for (int j = 0; j < m_obs.size(); j++) {
                id = m_obs[j].id;
                // id seems to be 1 based
                assoc_x = map_landmarks.landmark_list[id - 1].x_f;
                assoc_y = map_landmarks.landmark_list[id - 1].y_f;
                mx = m_obs[j].x;
                my = m_obs[j].y;
                // actually no need to calculate gauss_norm since weight doesn't need to be normalized
                //gauss_norm = 1 / (2 * stdx * stdy * M_PI);
                exponent = (mx - assoc_x) * (mx - assoc_x) / stdx_c;
                exponent += (my - assoc_y) * (my - assoc_y) / stdy_c;
                //weight *= gauss_norm * exp(-exponent);
                weight *= exp(-exponent);
            }

            particles[i].weight = weight;
            weights[i] = weight; 
        }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

        default_random_engine gen;
        discrete_distribution<int> disc_dist(weights.begin(), weights.end());
        vector<Particle> new_parts;
        int idx;

        for (int i = 0; i < num_particles; i++) {
            idx = disc_dist(gen);
            new_parts.push_back(particles[idx]);
        }
        particles = new_parts;
}

Particle ParticleFilter::SetAssociations(Particle &particle, std::vector<int> &associations, std::vector<double> &sense_x, std::vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	//particle.associations.clear();
	//particle.sense_x.clear();
	//particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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

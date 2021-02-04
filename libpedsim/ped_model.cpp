//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "ped_model.h"
#include "ped_waypoint.h"
#include "ped_model.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>

#include <stdlib.h>

void Ped::Model::setup(std::vector<Ped::Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario, IMPLEMENTATION implementation)
{
	// Convenience test: does CUDA work on this machine?
	cuda_test();

	// Set 
	agents = std::vector<Ped::Tagent*>(agentsInScenario.begin(), agentsInScenario.end());

	// Set up destinations
	destinations = std::vector<Ped::Twaypoint*>(destinationsInScenario.begin(), destinationsInScenario.end());

	// Sets the chosen implemenation. Standard in the given code is SEQ
	this->implementation = implementation;

	// Set up heatmap (relevant for Assignment 4)
	setupHeatmapSeq();
}


// Computes the agent positions for the agents between start and end in the array agents
void computeAgentPositions(int start, int end, std::vector<Ped::Tagent*> agents)
{
  for (start; start < end; start++) 
      {
	agents[start]->computeNextDesiredPosition();
	agents[start]->setX(agents[start]->getDesiredX());
	agents[start]->setY(agents[start]->getDesiredY());
      }
}


void *Ped::Model::preComputeFunc(Ped::Model::thread_info ti)
{
    int slice = ti.agents.size() / ti.num_threads;
    if (ti.num_threads == ti.thread_num + 1)
    {
      computeAgentPositions(slice * ti.thread_num, ti.agents.size(), ti.agents); 
    }
    else
    {
      computeAgentPositions(slice * ti.thread_num, slice * (ti.thread_num + 1), ti.agents);
    }

    return NULL;
}


void *thread_startup(void *thread_inf)
{
    Ped::Model::thread_info ti = *static_cast<Ped::Model::thread_info*>(thread_inf);
    Ped::Model test;
    test.preComputeFunc(ti);
    return NULL;
}


void Ped::Model::tick()
{
  // assuming threads between 2-8
  int num_threads = 2; //change this variable to chose number of threads we run on

  std::vector<Tagent*> agents = getAgents();
  switch(this->implementation){
  case SEQ:
    {
      computeAgentPositions(0, agents.size(), agents);
      break;
    }
  case OMP:
    {
      omp_set_num_threads(num_threads);
#pragma omp parallel for
      for (int i = 0; i < agents.size(); i++) 
	{
	  agents[i]->computeNextDesiredPosition();
	  agents[i]->setX(agents[i]->getDesiredX());
	  agents[i]->setY(agents[i]->getDesiredY());
	}
      break;
    }
  case PTHREAD:
    { 
      thread_info ti;
      pthread_t threads[num_threads];
      for (int i = 0; i < num_threads; i++)
	{
	  //std::cout << "\nloop: " << i;
	  ti.agents = agents;
	  ti.num_threads = num_threads;
	  ti.thread_num = i;
	  pthread_create(&threads[i], NULL, thread_startup, &ti);
	}

      for (int i = 0; i < num_threads; i++)
	{
	  pthread_join(threads[i], NULL);
	}
      
      break;
    }
  case CUDA:
    {
      break;
    }
  case VECTOR:
    {
      break;
    }
  }
}


  


////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent)
{
	// Search for neighboring agents
	set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2);

	// Retrieve their positions
	std::vector<std::pair<int, int> > takenPositions;
	for (std::set<const Ped::Tagent*>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt) {
		std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
		takenPositions.push_back(position);
	}

	// Compute the three alternative positions that would bring the agent
	// closer to his desiredPosition, starting with the desiredPosition itself
	std::vector<std::pair<int, int> > prioritizedAlternatives;
	std::pair<int, int> pDesired(agent->getDesiredX(), agent->getDesiredY());
	prioritizedAlternatives.push_back(pDesired);

	int diffX = pDesired.first - agent->getX();
	int diffY = pDesired.second - agent->getY();
	std::pair<int, int> p1, p2;
	if (diffX == 0 || diffY == 0)
	{
		// Agent wants to walk straight to North, South, West or East
		p1 = std::make_pair(pDesired.first + diffY, pDesired.second + diffX);
		p2 = std::make_pair(pDesired.first - diffY, pDesired.second - diffX);
	}
	else {
		// Agent wants to walk diagonally
		p1 = std::make_pair(pDesired.first, agent->getY());
		p2 = std::make_pair(agent->getX(), pDesired.second);
	}
	prioritizedAlternatives.push_back(p1);
	prioritizedAlternatives.push_back(p2);

	// Find the first empty alternative position
	for (std::vector<pair<int, int> >::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it) {

		// If the current position is not yet taken by any neighbor
		if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end()) {

			// Set the agent's position 
			agent->setX((*it).first);
			agent->setY((*it).second);

			break;
		}
	}
}

/// Returns the list of neighbors within dist of the point x/y. This
/// can be the position of an agent, but it is not limited to this.
/// \date    2012-01-29
/// \return  The list of neighbors
/// \param   x the x coordinate
/// \param   y the y coordinate
/// \param   dist the distance around x/y that will be searched for agents (search field is a square in the current implementation)
set<const Ped::Tagent*> Ped::Model::getNeighbors(int x, int y, int dist) const {

	// create the output list
	// ( It would be better to include only the agents close by, but this programmer is lazy.)	
	return set<const Ped::Tagent*>(agents.begin(), agents.end());
}

void Ped::Model::cleanup() {
	// Nothing to do here right now. 
}

Ped::Model::~Model()
{
	std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent){delete agent;});
	std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination){delete destination; });
}

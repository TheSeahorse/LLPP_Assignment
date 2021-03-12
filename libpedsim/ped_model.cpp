//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
//
// Adapted for Low Level Parallel Programming 2017
//
#include "heatmap_update.cuh"
#include "ped_model.h"
#include "ped_waypoint.h"
#include <iostream>
#include <stack>
#include <algorithm>
#include "cuda_testkernel.h"
#include <omp.h>
#include <thread>
#include <math.h>
#include <cstdio>
#include <utility>
#include "cuda.h"

#include <stdlib.h>

void Ped::Model::setup(std::vector<Ped::Tagent *> agentsInScenario, std::vector<Twaypoint *> destinationsInScenario, IMPLEMENTATION implementation)
{
  // Convenience test: does CUDA work on this machine?
  cuda_test();

  // Set up agents
  agents = std::vector<Ped::Tagent *>(agentsInScenario.begin(), agentsInScenario.end());

  // Set up destinations
  destinations = std::vector<Ped::Twaypoint *>(destinationsInScenario.begin(), destinationsInScenario.end());

  // Sets the chosen implemenation. Standard in the given code is SEQ
  this->implementation = implementation;

  // Set up heatmap (relevant for Assignment 4)
  setupHeatmapSeq();

  for (int i = 0; i < agents.size(); i++)
    {
      if (agents[i]->getX() < 80)
	{
	  if (agents[i]->getY() < 60)
	    {
	      this->agentsNW.push_back(agents[i]);
	    }
	  else
	    {
	      this->agentsSW.push_back(agents[i]);
	    }
	}
      else
	{
	  if (agents[i]->getY() < 60)
	    {
	      this->agentsNE.push_back(agents[i]);
	    }
	  else
	    {
	      this->agentsSE.push_back(agents[i]);
	    }
	}
    }
  int nr_agents = agents.size();
  nr_agents += (nr_agents % 4);
  this->agentX.resize(nr_agents);
  this->agentY.resize(nr_agents);
  this->destX.resize(nr_agents);
  this->destY.resize(nr_agents);
  this->destR.resize(nr_agents);

  for (int i = 0; i < agents.size(); i++)
    {
      // printf("size: %d, i:%d\n", nr_agents, i);
      agents[i]->getStartDestination();
	  agents[i]->computeNextDesiredPosition();

      this->agentX[i] = agents[i]->getX();
      this->agentY[i] = agents[i]->getY();
      this->destX[i] = agents[i]->destination->getx();
      this->destY[i] = agents[i]->destination->gety();
      this->destR[i] = agents[i]->destination->getr();
    }
}

// Computes the agent positions for the agents between start and end in the array agents
void computeAgentPositions(int start, int end, std::vector<Ped::Tagent *> agents)
{
  for (start; start < end; start++)
    {
      agents[start]->computeNextDesiredPosition();
      agents[start]->setX(agents[start]->getDesiredX());
      agents[start]->setY(agents[start]->getDesiredY());
    }
}

void Ped::Model::tick()
{
  // assuming threads between 2-8
  int num_threads = 8; //change this variable to chose number of threads we run on

  std::vector<Tagent *> agents = getAgents();
  switch (this->implementation)
    {
    case SEQ:
      {
	std::vector<Tagent *> tempAgent;
	for (int i = 0; i < agents.size(); i++)
	  {
	    agents[i]->computeNextDesiredPosition();
	    move(agents[i], agents, tempAgent);
	  }
		// tickTaskBased(num_threads);
	  	updateHeatmapSeq();
	//computeAgentPositions(0, agents.size(), agents);
	break;
      }
    case OMP:
      {
	omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(static)
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
	int num_agents = agents.size();
	int one_slice = num_agents / num_threads;

	/* Comment out the threads you're not using and add "num_agents" as the third argument
	   to the last thread you're using and make sure that the threads before that have 
	   one_slice*(thread_number-1) and one_slice*(thread_number) 
	   as their second and third arguments */
	std::thread first(computeAgentPositions, 0, one_slice, agents);
	std::thread second(computeAgentPositions, one_slice, one_slice * 2, agents);
	std::thread third(computeAgentPositions, one_slice * 2, one_slice * 3, agents);
	std::thread fourth(computeAgentPositions, one_slice * 3, num_agents, agents);
	/*
	  std::thread fifth(computeAgentPositions, one_slice*4, one_slice*5, agents);
	  std::thread sixth(computeAgentPositions, one_slice*5, one_slice*6, agents);
	  std::thread seventh(computeAgentPositions, one_slice*6, one_slice*7, agents);
	  std::thread eigth(computeAgentPositions, one_slice*7, num_agents, agents);
	*/
	first.join();
	second.join();
	third.join();
	fourth.join();
	/*
	  fifth.join();
	  sixth.join();
	  seventh.join();
	  eigth.join();
	*/
	break;
      }
    case CUDA:
      {
#pragma omp parallel
  {
#pragma omp single nowait
    {
#pragma omp task
      {
		updateHeatmapSeq();
	  }
		tickTaskBased(num_threads);
	#pragma omp taskwait
	}
	}
		break;
      }
    case VECTOR:
      {
	int i;
	int j;
	omp_set_num_threads(num_threads);
#pragma omp parallel for private(i)
	for (i = 0; i < agents.size(); i++)
	  {
	    agents[i]->setX((int)round(this->agentX[i]));
	    agents[i]->setY((int)round(this->agentY[i]));

	    Twaypoint *newDest = agents[i]->getNextDestination();
	    this->destX[i] = newDest->getx();
	    this->destY[i] = newDest->gety();
	  }

#pragma omp simd
	for (j = 0; j < agents.size(); j += 4)
	  {
	    this->x = _mm_load_ps(&this->agentX[j]);
	    this->y = _mm_load_ps(&this->agentY[j]);
	    this->diffX = _mm_load_ps(&this->destX[j]);
	    this->diffY = _mm_load_ps(&this->destY[j]);

	    this->diffX = _mm_sub_ps(this->diffX, this->x);
	    this->diffY = _mm_sub_ps(this->diffY, this->y);

	    this->sqrX = _mm_mul_ps(this->diffX, this->diffX);
	    this->sqrY = _mm_mul_ps(this->diffY, this->diffY);

	    this->sumSqr = _mm_add_ps(this->sqrX, this->sqrY);
	    this->len = _mm_sqrt_ps(this->sumSqr);

	    this->desPosX = _mm_div_ps(this->diffX, this->len);
	    this->desPosY = _mm_div_ps(this->diffY, this->len);

	    this->desPosX = _mm_add_ps(this->x, this->desPosX);
	    this->desPosY = _mm_add_ps(this->y, this->desPosY);

	    _mm_store_ps(&this->agentX[j], this->desPosX);
	    _mm_store_ps(&this->agentY[j], this->desPosY);
	  }
	break;
      }
    case TASK:
      {
	tickTaskBased(num_threads);
      }
    }
}


void Ped::Model::tickTaskBased(int num_threads)
{
	// printf("Starting tickTaskBased\n");
  omp_set_num_threads(num_threads);
#pragma omp parallel
  {
#pragma omp single nowait
    {
#pragma omp task
      {
	for (int i = 0; i < this->agentsSW.size(); i++)
	  {
	    this->agentsSW[i]->computeNextDesiredPosition();
	    if(checkPosition(agentsSW[i]))
	      {
		this->tempSW.push_back(this->agentsSW[i]);
		this->agentsSW.erase(this->agentsSW.begin() + i);
		i--;
	      }
	    else 
	      {
		move(this->agentsSW[i], this->agentsSW, this->tempSW);
	      }
	  }
      }
#pragma omp task
      {
	for (int i = 0; i < this->agentsNW.size(); i++)
	  {
	    this->agentsNW[i]->computeNextDesiredPosition();
	    if(checkPosition(agentsNW[i]))
	      {
		this->tempNW.push_back(this->agentsNW[i]);
		this->agentsNW.erase(this->agentsNW.begin() + i);
		i--;
	      }
	    else 
	      {
		move(this->agentsNW[i], this->agentsNW, this->tempNW);
	      }
	  }
      }
#pragma omp task
      {
	for (int i = 0; i < this->agentsSE.size(); i++)
	  {
	    this->agentsSE[i]->computeNextDesiredPosition();
	    if(checkPosition(agentsSE[i]))
	      {
		this->tempSE.push_back(this->agentsSE[i]);
		this->agentsSE.erase(this->agentsSE.begin() + i);
		i--;
	      }
	    else 
	      {
		move(this->agentsSE[i], this->agentsSE, this->tempSE);
	      }
	  }
      }

	for (int i = 0; i < this->agentsNE.size(); i++)
	  {
	    this->agentsNE[i]->computeNextDesiredPosition();
	    if(checkPosition(agentsNE[i]))
	      {
		this->tempNE.push_back(this->agentsNE[i]);
		this->agentsNE.erase(this->agentsNE.begin() + i);
		i--;
	      }
	    else 
	      {
		move(this->agentsNE[i], this->agentsNE, this->tempNE);
	      }
      }
    }
#pragma omp taskwait


  }
//   updateHeatmapSeq();
  int largestArray = (int)std::max(std::max(this->tempSW.size(),this->tempNW.size()),std::max(this->tempSE.size(),this->tempNE.size()));

  std::vector<Ped::Tagent *> allTemps;
  allTemps.insert(allTemps.end(), this->tempSE.begin(), this->tempSE.end());
  allTemps.insert(allTemps.end(), this->tempNE.begin(), this->tempNE.end());
  allTemps.insert(allTemps.end(), this->tempSW.begin(), this->tempSW.end());
  allTemps.insert(allTemps.end(), this->tempNW.begin(), this->tempNW.end());

  for(int i=0; i < largestArray; i++)
    {

      if(i < this->tempSW.size())
	{
	  computeAndMove(this->tempSW[i], this->agentsSW, allTemps);
	}
	
      if(i < this->tempNW.size())
	{
	  computeAndMove(this->tempNW[i], this->agentsNW, allTemps);
	}
		
      if(i < this->tempSE.size())
	{
	  computeAndMove(this->tempSE[i], this->agentsSE, allTemps);
	}
		
      if(i < this->tempNE.size())
	{
	  computeAndMove(this->tempNE[i], this->agentsNE, allTemps);
	}

    }

  this->tempSE.clear();
  this->tempNE.clear();
  this->tempSW.clear();
  this->tempNW.clear();   
//   printf("Finished tickTaskBased\n"); 
}


////////////
/// Everything below here relevant for Assignment 3.
/// Don't use this for Assignment 1!
///////////////////////////////////////////////

void Ped::Model::computeAndMove(Ped::Tagent *agent, std::vector<Ped::Tagent *> agentVector, std::vector<Ped::Tagent *> allTemps)
{
  move(agent, agentVector, allTemps);
  moveAgentToArray(agent);
}

bool Ped::Model::checkPosition(Ped::Tagent *agent)
{
  if((agent->getDesiredX() >= 78 and
      agent->getDesiredX() < 82) or
     (agent->getDesiredY() >= 58 and
      agent->getDesiredY() < 62))
    {
      return true;
    }
  else
    {
      return false;
    }
}

void Ped::Model::moveAgentToArray(Ped::Tagent *agent)
{

  if (agent->getX() < 80)
    {
      if (agent->getY() < 60)
	{
	  agentsNW.push_back(std::move(agent));
	}
      else
	{
	  agentsSW.push_back(std::move(agent));
	}
    }
  else if (agent->getX() >= 80)
    {
      if (agent->getY() < 60)
	{
	  agentsNE.push_back(std::move(agent));
	}
      else
	{
	  agentsSE.push_back(std::move(agent));
	}
    }
  else
    {
      printf("Somethings wrong!\n");
    }

}

// Moves the agent to the next desired position. If already taken, it will
// be moved to a location close to it.
void Ped::Model::move(Ped::Tagent *agent, std::vector<Ped::Tagent *> agentVector, std::vector<Ped::Tagent *> tempVector)
{
  // Search for neighboring agents
  set<const Ped::Tagent *> neighbors = getNeighbors(agent->getX(), agent->getY(), 2, agentVector, tempVector);

  // Retrieve their positions
  std::vector<std::pair<int, int>> takenPositions;
  for (std::set<const Ped::Tagent *>::iterator neighborIt = neighbors.begin(); neighborIt != neighbors.end(); ++neighborIt)
    {
      std::pair<int, int> position((*neighborIt)->getX(), (*neighborIt)->getY());
      takenPositions.push_back(position);
    }

  // Compute the three alternative positions that would bring the agent
  // closer to his desiredPosition, starting with the desiredPosition itself
  std::vector<std::pair<int, int>> prioritizedAlternatives;
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
  else
    {
      // Agent wants to walk diagonally
      p1 = std::make_pair(pDesired.first, agent->getY());
      p2 = std::make_pair(agent->getX(), pDesired.second);
    }
  prioritizedAlternatives.push_back(p1);
  prioritizedAlternatives.push_back(p2);

  // Find the first empty alternative position
  for (std::vector<pair<int, int>>::iterator it = prioritizedAlternatives.begin(); it != prioritizedAlternatives.end(); ++it)
    {

      // If the current position is not yet taken by any neighbor
      if (std::find(takenPositions.begin(), takenPositions.end(), *it) == takenPositions.end())
	{

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
set<const Ped::Tagent *> Ped::Model::getNeighbors(int x, int y, int dist, std::vector<Ped::Tagent *> agentVector, std::vector<Ped::Tagent *> tempVector) const
{

  std::vector<Ped::Tagent *> neighbors(0);

  //   for (int i = 0; i < agents.size(); i++)
  for (int i = 0; i < agentVector.size(); i++)
    {
      int aX = agentVector[i]->getX();
      int aY = agentVector[i]->getY();

      if (aX < (x + dist) and
	  aX > (x - dist) and
	  aY < (y + dist) and
	  aY > (y - dist) and
	  (aX != x or aY != y))
	{
	  neighbors.push_back(agentVector[i]);
	}
    }

  for (int i = 0; i < tempVector.size(); i++)
    {
      int aX = tempVector[i]->getX();
      int aY = tempVector[i]->getY();

      if (aX < (x + dist) and
	  aX > (x - dist) and
	  aY < (y + dist) and
	  aY > (y - dist) and
	  (aX != x or aY != y))
	{
	  neighbors.push_back(tempVector[i]);
	}
    }

  // create the output list
  // ( It would be better to include only the agents close by, but this programmer is lazy.)
  //   std::cout << "amount of neighbors: " << neighbors.size() << "\n";
  return set<const Ped::Tagent *>(neighbors.begin(), neighbors.end());
}

void Ped::Model::cleanup()
{
  // Nothing to do here right now.
}

Ped::Model::~Model()
{
  std::for_each(agents.begin(), agents.end(), [](Ped::Tagent *agent) { delete agent; });
  std::for_each(destinations.begin(), destinations.end(), [](Ped::Twaypoint *destination) { delete destination; });
}

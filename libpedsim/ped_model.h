//
// pedsim - A microscopic pedestrian simulation system.
// Copyright (c) 2003 - 2014 by Christian Gloor
//
// Adapted for Low Level Parallel Programming 2017
//
// Model coordinates a time step in a scenario: for each
// time step all agents need to be moved by one position if
// possible.
//
#ifndef _ped_model_h_
#define _ped_model_h_

#include <vector>
#include <map>
#include <set>
#include <emmintrin.h>
#include "ped_agent.h"

namespace Ped{
	class Tagent;

	// The implementation modes for Assignment 1 + 2:
	// chooses which implementation to use for tick()
	enum IMPLEMENTATION { CUDA, VECTOR, OMP, PTHREAD, SEQ, TASK };

	class Model
	{
	public:
	  __m128 x, y, r, diffX, diffY, sqrX, sqrY, sumSqr, len, desPosX, desPosY, newDestBool;
	  // Sets everything up
	  void setup(std::vector<Tagent*> agentsInScenario, std::vector<Twaypoint*> destinationsInScenario,IMPLEMENTATION implementation);
	        
	  // Coordinates a time step in the scenario: move all agents by one step (if applicable).
	  void tick();

	  // Returns the agents of this scenario
	  const std::vector<Tagent*> getAgents() const { return agents; };
	  
	  std::vector<float> agentX;
	  std::vector<float> agentY;
	  std::vector<float> destX;
	  std::vector<float> destY;
	  std::vector<float> destR;
	  std::vector<float> reachedDest;
	  
	  //const std::vector<float> getAgentX() const {return agentX;};
	  //const std::vector<float> getAgentY() const {return agentY;};
	  //const std::vector<float> getDestX() const {return destX;};
	  //const std::vector<float> getDestY() const {return destY;};
	  // Adds an agent to the tree structure
	  void placeAgent(const Ped::Tagent *a);
	  
	  void tickTaskBased(int num_threads);

	  // Moves an agent to the right array if they've crossed.
	  void moveAgentToArray(Ped::Tagent *agent);

	  void computeAndMove(Ped::Tagent *agents, std::vector<Ped::Tagent *> agentVector, std::vector<Ped::Tagent *> allTemps);

	  bool checkPosition(Ped::Tagent *agent);

	  // Cleans up the tree and restructures it. Worth calling every now and then.
	  void cleanup();
	  ~Model();

	  // Returns the heatmap visualizing the density of agents
	  int const * const * getHeatmap() const { return blurred_heatmap; };
	  int getHeatmapSize() const;

	private:

	  // Denotes which implementation (sequential, parallel implementations..)
		// should be used for calculating the desired positions of
		// agents (Assignment 1)
		IMPLEMENTATION implementation;
		int num_threads;
		// The agents in this scenario
		std::vector<Tagent*> agents;
		
		std::vector<Tagent*> agentsSW;
		std::vector<Tagent*> agentsNW;
		std::vector<Tagent*> agentsSE;
		std::vector<Tagent*> agentsNE;

		std::vector<Tagent*> tempSW;
		std::vector<Tagent*> tempNW;
		std::vector<Tagent*> tempSE;
		std::vector<Tagent*> tempNE;

		// The waypoints in this scenario
		std::vector<Twaypoint*> destinations;

		// Moves an agent towards its next position
		void move(Ped::Tagent *agent, std::vector<Ped::Tagent *> agentVector, std::vector<Ped::Tagent *> tempVector);
		
		
		////////////
		/// Everything below here won't be relevant until Assignment 3
		///////////////////////////////////////////////

		// Returns the set of neighboring agents for the specified position
		set<const Ped::Tagent*> getNeighbors(int x, int y, int dist, std::vector<Ped::Tagent *> agentVector, std::vector<Ped::Tagent *> tempVector) const;

		////////////
		/// Everything below here won't be relevant until Assignment 4
		///////////////////////////////////////////////

#define SIZE 1024
#define CELLSIZE 5
#define SCALED_SIZE SIZE*CELLSIZE

		// The heatmap representing the density of agents
		int ** heatmap;

		// The scaled heatmap that fits to the view
		int ** scaled_heatmap;

		// The final heatmap: blurred and scaled to fit the view
		int ** blurred_heatmap;

		int *hm;
		int *shm;
		int *bhm;

		// void setupHeatmap();
		void setupHeatmapSeq();
		// void updateHeatmap(int **heatmap, int **scaled_heatmap, int **blurred_heatmap);
		void updateHeatmapSeq();
	};
}
#endif

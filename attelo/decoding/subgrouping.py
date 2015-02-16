class AstarStrategy(Enum):
    """
    What sort of  to apply during decoding:

        * simple:
        * intra: intra-sentence level first 
    """
    simple = 1
    intra_heads = 2
    intra_last = 3
    intra_only = 4
    intra_rfc = 5

    @classmethod
    def from_string(cls, string):
        "command line arg to AstarStrategy"
        names = {x.name: x for x in cls}
        strat = names.get(string)
        if strat is not None:
            return strat
        else:
            oops = "invalid choice: {}, choose from {}"
            choices = ", ".join('{}'.format(x) for x in names)
            raise ArgumentTypeError(oops.format(string, choices))

_intra_strategies = frozenset((AstarStrategy.intra_heads, 
                               AstarStrategy.intra_last,
                               AstarStrategy.intra_only,
                               AstarStrategy.intra_rfc
                               ))

def find_head_of_tree(edge_list):
    """find the head of a tree given as a list of edge (only node appearing as target only)"""
    #print(edge_list, file=sys.stderr)
    all = dict(((e2,e1) for (e1,e2,r) in edge_list))
    sources = frozenset(all.values())
    targets = frozenset(all.keys())
    head = sources - targets 
    if len(head)!=1: 
        print("wrong graph (not a tree) appearing in prediction", file = sys.stderr)
        print( edge_list, file = sys.stderr)
        print( head, file = sys.stderr)
        sys.exit(0)
    else: 
        head = list(head)[0]
        print("head = %s"%head,file=sys.stderr)
        return head
    

def same_sentence(e1id,e2id,edus):
    return edus[e1id].subgrouping == edus[e2id].subgrouping



def tmp(self):
        if self._args.strategy in _intra_strategies:
            # launch a decoder per sentence
            #   - sent parses collect separate parses
            #   - to_link will collect admissible edus for document level attachmt 
            #     (head parses for now=first edu of sentence)

            #   - accessible will collect sentence sub-edus that are legit as targets for attachmts but are not
            #     to be linked (eg, last edu of a sentence, or right-frontier-constraint
            sent_parses = []
            to_link = []
            accessible = []
            for (i,sent) in enumerate(order_by_sentence(sorted_edus)):
                print("doing sentence %d, with %d nodes"%(i+1,len(sent)),file=sys.stderr)
                if len(sent)==1:
                    head = sent[0]
                else:
                    if (self._args.beam):
                        astar = DiscourseBeamSearch(heuristic=heuristic,shared=search_shared,queue_size=self._args.beam)
                    else:
                        astar = DiscourseSearch(heuristic=heuristic,shared=search_shared)
                    genall = astar.launch(DiscData(accessible=[], tolink=sent),norepeat=True, verbose=False)
                    endstate = genall.next()
                    sol = astar.recover_solution(endstate)
                    sent_parses.extend(sol)
                    head = find_head_of_tree(sol)
                ##########

                to_link.append(head)

                if self._args.strategy==AstarStrategy.intra_heads:
                    pass
                    #accessible.append(find_head_of_tree(sol))
                elif self._args.strategy==AstarStrategy.intra_last:
                     accessible.extend(list(set([head,sent[-1]])))
                elif self._args.strategy==AstarStrategy.intra_rfc:
                    # TODO: check that last edu is in it ...
                    if len(sent)>1:
                        accessible.extend(endstate.data().accessible())

        # this should be ventilated above. here for easier testing for now
        if self._args.strategy==AstarStrategy.intra_only:
            return [sent_parses]
        if self._args.strategy  in _intra_strategies: 
            # recombine sub parses:
            print("start document decoding with intra/inter sentence model with strategy =%s "%self._args.strategy, file=sys.stderr)
            if (self._args.beam):
                astar = DiscourseBeamSearch(heuristic=heuristic, shared=search_shared, queue_size=self._args.beam)
            else:
                astar = DiscourseSearch(heuristic=heuristic, shared=search_shared)
            #genall = astar.launch(DiscData(accessible=[to_link[0]], tolink=to_link[1:]),
            # sentence heads + RFC accessbile 
            # FIXME: this is wrong for everything but intra_heads where accessible should be empty
            # 
            #print("tolink",to_link,file=sys.stderr)
            #print("access",[fake_root]+accessible,file=sys.stderr)
            #genall = astar.launch(DiscData(accessible=["ROOT"]+accessible, tolink=to_link), norepeat=True, verbose=False)
            if len(to_link)<2: 
                print("error document has too few edus ??", to_link, file=sys.stderr)
                all_solutions = [sent_parses]
            else:
                genall = astar.launch(DiscData(accessible=[to_link[0]], tolink=to_link[1:]), norepeat=True, verbose=False, filter_states = lambda x, (y,z): not(same_sentence(y,z,x["edus"])))
                endstate = genall.next()
                sol = astar.recover_solution(endstate)
                all_solutions = [sol+sent_parses]


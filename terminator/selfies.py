"""
This is the selfies decoder as implemented in v1.0.4.
Taken from: https://github.com/aspuru-guzik-group/selfies
"""

from collections import OrderedDict
from itertools import product
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

ATOM_TYPE = 1
BRANCH_TYPE = 2
RING_TYPE = 3


default_bond_constraints = {
    'H': 1,
    'F': 1,
    'Cl': 1,
    'Br': 1,
    'I': 1,
    'O': 2,
    'O+1': 3,
    'O-1': 1,
    'N': 3,
    'N+1': 4,
    'N-1': 2,
    'C': 4,
    'C+1': 5,
    'C-1': 3,
    'P': 5,
    'P+1': 6,
    'P-1': 4,
    'S': 6,
    'S+1': 7,
    'S-1': 5,
    '?': 8,
}

octet_rule_bond_constraints = dict(default_bond_constraints)
octet_rule_bond_constraints.update(
    {'S': 2, 'S+1': 3, 'S-1': 1, 'P': 3, 'P+1': 4, 'P-1': 2}
)

hypervalent_bond_constraints = dict(default_bond_constraints)
hypervalent_bond_constraints.update({'Cl': 7, 'Br': 7, 'I': 7, 'N': 5})

_bond_constraints = default_bond_constraints


def get_semantic_robust_alphabet() -> Set[str]:
    """Returns a subset of all symbols that are semantically constrained
    by :mod:`selfies`.
    These semantic constraints can be configured with
    :func:`selfies.set_semantic_constraints`.
    :return: a subset of all symbols that are semantically constrained.
    """

    alphabet_subset = set()

    organic_subset = {'B', 'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'}
    bonds = {'': 1, '=': 2, '#': 3}

    # add atomic symbols
    for (a, c), (b, m) in product(_bond_constraints.items(), bonds.items()):

        if (m > c) or (a == '?'):
            continue

        if a in organic_subset:
            symbol = "[{}{}]".format(b, a)
        else:
            symbol = "[{}{}expl]".format(b, a)

        alphabet_subset.add(symbol)

    # add branch and ring symbols
    for i in range(1, 4):
        alphabet_subset.add("[Ring{}]".format(i))
        alphabet_subset.add("[Expl=Ring{}]".format(i))

        for j in range(1, 4):
            alphabet_subset.add("[Branch{}_{}]".format(i, j))

    return alphabet_subset


def get_default_constraints() -> Dict[str, int]:
    """Returns the preset "default" bond constraint settings.
    :return: the default constraint settings.
    """

    global default_bond_constraints
    return dict(default_bond_constraints)


def get_octet_rule_constraints() -> Dict[str, int]:
    """Returns the preset "octet rule" bond constraint settings. These
    constraints are a harsher version of the default constraints, so that
    the `octet rule <https://en.wikipedia.org/wiki/Octet_rule>`_
    is obeyed. In particular, ``S`` and ``P`` are
    restricted to a 2 and 3 bond capacity, respectively (and similarly with
    ``S+``, ``S-``, ``P+``, ``P-``).
    :return: the octet rule constraint settings.
    """

    global octet_rule_bond_constraints
    return dict(octet_rule_bond_constraints)


def get_hypervalent_constraints() -> Dict[str, int]:
    """Returns the preset "hypervalent" bond constraint settings. These
    constraints are a relaxed version of the default constraints, to allow
    for `hypervalent molecules
    <https://en.wikipedia.org/wiki/Hypervalent_molecule>`_.
    In particular, ``Cl``, ``Br``, and ``I``
    are relaxed to a 7 bond capacity, and ``N`` is relaxed to a 5 bond
    capacity.
    :return: the hypervalent constraint settings.
    """

    global hypervalent_bond_constraints
    return dict(hypervalent_bond_constraints)


def get_semantic_constraints() -> Dict[str, int]:
    """Returns the semantic bond constraints that :mod:`selfies` is currently
    operating on.
    Returned is the argument of the most recent call of
    :func:`selfies.set_semantic_constraints`, or the default bond constraints
    if the function has not been called yet. Once retrieved, it is copied and
    then returned. See :func:`selfies.set_semantic_constraints` for further
    explanation.
    :return: the bond constraints :mod:`selfies` is currently operating on.
    """

    global _bond_constraints
    return dict(_bond_constraints)


def set_semantic_constraints(bond_constraints: Optional[Dict[str, int]] = None) -> None:
    """Configures the semantic constraints of :mod:`selfies`.
    The SELFIES grammar is enforced dynamically from a dictionary
    ``bond_constraints``. The keys of the dictionary are atoms and/or ions
    (e.g. ``I``, ``Fe+2``). To denote an ion, use the format ``E+C``
    or ``E-C``, where ``E`` is an element and ``C`` is a positive integer.
    The corresponding value is the maximum number of bonds that atom or
    ion can make, between 1 and 8 inclusive. For example, one may have:
        * ``bond_constraints['I'] = 1``
        * ``bond_constraints['C'] = 4``
    :func:`selfies.decoder` will only generate SMILES that respect the bond
    constraints specified by the dictionary. In the example above, both
    ``'[C][=I]'`` and ``'[I][=C]'`` will be translated to ``'CI'`` and
    ``'IC'`` respectively, because ``I`` has been configured to make one bond
    maximally.
    If an atom or ion is not specified in ``bond_constraints``, it will
    by default be constrained to 8 bonds. To change the default setting
    for unrecognized atoms or ions, set ``bond_constraints['?']`` to the
    desired integer (between 1 and 8 inclusive).
    :param bond_constraints: a dictionary representing the semantic
        constraints the updated SELFIES will operate upon. Defaults to
        ``None``; in this case, a default dictionary will be used.
    :return: ``None``.
    """

    global _bond_constraints

    if bond_constraints is None:
        _bond_constraints = default_bond_constraints

    else:

        # error checking
        if '?' not in bond_constraints:
            raise ValueError("bond_constraints missing '?' as a key.")

        for key, value in bond_constraints.items():
            if not (1 <= value <= 8):
                raise ValueError(
                    "bond_constraints['{}'] not between "
                    "1 and 8 inclusive.".format(key)
                )

        _bond_constraints = dict(bond_constraints)


# Symbol State Dict Functions ==============================================


def get_next_state(symbol: str, state: int) -> Tuple[str, int]:
    """Enforces the grammar rules for standard SELFIES symbols.
    Given the current non-branch, non-ring symbol and current derivation
    state, retrieves the derived SMILES symbol and the next derivation
    state.
    :param symbol: a SELFIES symbol that is not a Ring or Branch.
    :param state: the current derivation state.
    :return: a tuple of (1) the derived symbol, and
        (2) the next derivation state.
    """

    if symbol == '[epsilon]':
        return ('', 0) if state == 0 else ('', -1)

    # convert to smiles symbol
    bond = ''
    if symbol[1] in {'/', '\\', '=', '#'}:
        bond = symbol[1]
    bond_num = get_num_from_bond(bond)

    if symbol[-5:] == 'expl]':  # e.g. [C@@Hexpl]
        smiles_symbol = "[{}]".format(symbol[1 + len(bond) : -5])
    else:
        smiles_symbol = symbol[1 + len(bond) : -1]

    # get bond capacity
    element, h_count, charge = parse_atom_symbol(smiles_symbol)

    if charge == 0:
        atom_or_ion = element
    else:
        atom_or_ion = "{}{:+}".format(element, charge)

    max_bonds = _bond_constraints.get(atom_or_ion, _bond_constraints['?'])

    if (h_count > max_bonds) or (h_count == max_bonds and state > 0):
        raise ValueError(
            "too many Hs in symbol '{}'; consider "
            "adjusting bond constraints".format(symbol)
        )
    max_bonds -= h_count  # hydrogens consume 1 bond

    # calculate next state
    if state == 0:
        bond = ''
        next_state = max_bonds
    else:
        if bond_num > min(state, max_bonds):
            bond_num = min(state, max_bonds)
            bond = get_bond_from_num(bond_num)

        next_state = max_bonds - bond_num
        if next_state == 0:
            next_state = -1

    return (bond + smiles_symbol), next_state


# Branch State Dict Functions =================================================


def get_next_branch_state(branch_symbol: str, state: int) -> Tuple[int, int]:
    """Enforces the grammar rules for SELFIES Branch symbols.
    Given the branch symbol and current derivation state, retrieves
    the initial branch derivation state (i.e. the derivation state that the
    new branch begins on), and the next derivation state (i.e. the derivation
    state after the branch is created).
    :param branch_symbol: the branch symbol (e.g. [Branch1_2], [Branch3_1])
    :param state: the current derivation state.
    :return: a tuple of (1) the initial branch state, and
        (2) the next derivation state.
    """

    branch_type = int(branch_symbol[-2])  # branches of the form [BranchL_X]

    if not (1 <= branch_type <= 3):
        raise ValueError("unknown branch symbol '{}'".format(branch_symbol))

    if 2 <= state <= 8:
        branch_init_state = min(state - 1, branch_type)
        next_state = state - branch_init_state
        return branch_init_state, next_state
    else:
        return -1, state


# SELFIES Symbol to N Functions ============================================

_index_alphabet = [
    '[C]',
    '[Ring1]',
    '[Ring2]',
    '[Branch1_1]',
    '[Branch1_2]',
    '[Branch1_3]',
    '[Branch2_1]',
    '[Branch2_2]',
    '[Branch2_3]',
    '[O]',
    '[N]',
    '[=N]',
    '[=C]',
    '[#C]',
    '[S]',
    '[P]',
]

# _alphabet_code takes as a key a SELFIES symbol, and its corresponding value
# is the index of the key.

_alphabet_code = {c: i for i, c in enumerate(_index_alphabet)}


def get_n_from_symbols(*symbols: List[str]) -> int:
    """Computes N from a list of SELFIES symbols.
    Converts a list of SELFIES symbols [c_1, ..., c_n] into a number N.
    This is done by converting each symbol c_n to an integer idx(c_n) via
    ``_alphabet_code``, and then treating the list as a number in base
    len(_alphabet_code). If a symbol is unrecognized, it is given value 0 by
    default.
    :param symbols: a list of SELFIES symbols.
    :return: the corresponding N for ``symbols``.
    """

    N = 0
    for i, c in enumerate(reversed(symbols)):
        N_i = _alphabet_code.get(c, 0) * (len(_alphabet_code) ** i)
        N += N_i
    return N


def get_symbols_from_n(n: int) -> List[str]:
    """Converts an integer n into a list of SELFIES symbols that, if
    passed into ``get_n_from_symbols`` in that order, would have produced n.
    :param n: an integer from 0 to 4095 inclusive.
    :return: a list of SELFIES symbols representing n in base
        ``len(_alphabet_code)``.
    """

    if n == 0:
        return [_index_alphabet[0]]

    symbols = []
    base = len(_index_alphabet)
    while n:
        symbols.append(_index_alphabet[n % base])
        n //= base
    return symbols[::-1]


# Helper Functions ============================================================


def get_num_from_bond(bond_symbol: str) -> int:
    """Retrieves the bond multiplicity from a SMILES symbol representing
    a bond. If ``bond_symbol`` is not known, 1 is returned by default.
    :param bond_symbol: a SMILES symbol representing a bond.
    :return: the bond multiplicity of ``bond_symbol``, or 1 if
        ``bond_symbol`` is not recognized.
    """

    if bond_symbol == "=":
        return 2
    elif bond_symbol == "#":
        return 3
    else:
        return 1


def get_bond_from_num(n: int) -> str:
    """Returns the SMILES symbol representing a bond with multiplicity
    ``n``. More specifically, ``'' = 1`` and ``'=' = 2`` and ``'#' = 3``.
    :param n: either 1, 2, 3.
    :return: the SMILES symbol representing a bond with multiplicity ``n``.
    """

    return ('', '=', '#')[n - 1]


def find_element(atom_symbol: str) -> Tuple[int, int]:
    """Returns the indices of the element component of a SMILES atom symbol.
    That is, if atom_symbol[i:j] is the element substring of the SMILES atom,
    then (i, j) is returned. For example:
        *   _find_element('b') = (0, 1).
        *   _find_element('B') = (0, 1).
        *   _find_element('[13C]') = (3, 4).
        *   _find_element('[nH+]') = (1, 2).
    :param atom_symbol: a SMILES atom.
    :return: a tuple of the indices of the element substring of
        ``atom_symbol``.
    """

    if atom_symbol[0] != '[':
        return 0, len(atom_symbol)

    i = 1
    while atom_symbol[i].isdigit():  # skip isotope number
        i += 1

    if atom_symbol[i + 1].isalpha() and atom_symbol[i + 1] != 'H':
        return i, i + 2
    else:
        return i, i + 1


def parse_atom_symbol(atom_symbol: str) -> Tuple[str, int, int]:
    """Parses a SMILES atom symbol and returns its element component,
    number of associated hydrogens, and charge.
    See http://opensmiles.org/opensmiles.html for the formal grammar
    of SMILES atom symbols. Note that only @ and @@ are currently supported
    as chiral specifications.
    :param atom_symbol: a SMILES atom symbol.
    :return: a tuple of (1) the element of ``atom_symbol``, (2) the hydrogen
        count, and (3) the charge.
    """

    if atom_symbol[0] != '[':
        return atom_symbol, 0, 0

    atom_start, atom_end = find_element(atom_symbol)
    i = atom_end

    # skip chirality
    if atom_symbol[i] == '@':  # e.g. @
        i += 1
    if atom_symbol[i] == '@':  # e.g. @@
        i += 1

    h_count = 0  # hydrogen count
    if atom_symbol[i] == 'H':
        h_count = 1

        i += 1
        if atom_symbol[i].isdigit():  # e.g. [CH2]
            h_count = int(atom_symbol[i])
            i += 1

    charge = 0  # charge count
    if atom_symbol[i] in ('+', '-'):
        charge = 1 if atom_symbol[i] == '+' else -1

        i += 1
        if atom_symbol[i] in ('+', '-'):  # e.g. [Cu++]
            while atom_symbol[i] in ('+', '-'):
                charge += 1 if atom_symbol[i] == '+' else -1
                i += 1

        elif atom_symbol[i].isdigit():  # e.g. [Cu+2]
            s = i
            while atom_symbol[i].isdigit():
                i += 1
            charge *= int(atom_symbol[s:i])

    return atom_symbol[atom_start:atom_end], h_count, charge


def kekulize_parser(
    smiles_gen: Iterable[Tuple[str, str, int]]
) -> Iterable[Tuple[str, str, int]]:
    """Kekulizes a SMILES in the form of an iterable.
    This method intercepts the output of ``encoder._parse_smiles``, and
    acts as filter that kekulizes the SMILES. The motivation for having
    this setup is that string parsing and concatenation is minimized,
    as the parsing is already done by ``_parse_smiles``.
    Reference: https://depth-first.com/articles/2020/02/10/a-comprehensive
               -treatment-of-aromaticity-in-the-smiles-language/
    :param smiles_gen: an iterator returned by ``encoder._parse_smiles``.
    :return: an iterator representing the kekulized SMILES, in the same
        format as that returned by ``encoder._parse_smiles``.
    """

    # save to list, so the iterator can be used across multiple functions
    # change elements from tuple -> list to allow in-place modifications
    smiles_symbols = list(map(list, smiles_gen))

    mol_graph = MolecularGraph(smiles_symbols)

    rings = {}
    _build_molecular_graph(mol_graph, smiles_symbols, rings)

    if mol_graph.aro_indices:
        _kekulize(mol_graph)

    for x in mol_graph.smiles_symbols:  # return as iterator
        yield tuple(x)


def _build_molecular_graph(
    graph,
    smiles_symbols: List[List[Union[str, int]]],
    rings: Dict[int, Tuple[int, int]],
    prev_idx: int = -1,
    curr_idx: int = -1,
) -> int:
    """From the iterator returned by ``encoder._parse_smiles``, builds
    a graph representation of the molecule.
    This is done by iterating through ``smiles_symbols``, and then adding bonds
    to the molecular graph. Note that ``smiles_symbols`` is mutated in this
    method, for convenience.
    :param graph: the MolecularGraph to be added to.
    :param smiles_symbols: a list created from the iterator returned
        by ``encoder._parse_smiles``.
    :param rings: an, initially, empty dictionary used to keep track of
        rings to be made.
    :param prev_idx:
    :param curr_idx:
    :return: the last index of ``smiles_symbols`` that was processed.
    """

    while curr_idx + 1 < len(smiles_symbols):

        curr_idx += 1
        _, symbol, symbol_type = smiles_symbols[curr_idx]

        if symbol_type == ATOM_TYPE:
            if prev_idx >= 0:
                graph.add_bond(prev_idx, curr_idx, curr_idx)
            prev_idx = curr_idx

        elif symbol_type == BRANCH_TYPE:
            if symbol == '(':
                curr_idx = _build_molecular_graph(
                    graph, smiles_symbols, rings, prev_idx, curr_idx
                )
            else:
                break

        else:
            if symbol in rings:
                left_idx, left_bond_idx = rings.pop(symbol)
                right_idx, right_bond_idx = prev_idx, curr_idx

                # we mutate one bond index to be '', so that we
                # can faithfully represent the bond to be localized at
                # one index. For example, C=1CCCC=1 --> C1CCCC=1.

                if smiles_symbols[left_bond_idx][0] != '':
                    bond_idx = left_bond_idx
                    smiles_symbols[right_bond_idx][0] = ''
                else:
                    bond_idx = right_bond_idx
                    smiles_symbols[left_bond_idx][0] = ''

                graph.add_bond(left_idx, right_idx, bond_idx)
            else:
                rings[symbol] = (prev_idx, curr_idx)

    return curr_idx


def _kekulize(mol_graph) -> None:
    """Kekulizes the molecular graph.
    :param mol_graph: a molecular graph to be kekulized.
    :return: None.
    """

    mol_graph.prune_to_pi_subgraph()

    visited = set()
    for i in mol_graph.get_nodes_by_num_edges():
        success = mol_graph.dfs_assign_bonds(i, visited, set(), set())
        if not success:
            raise ValueError("kekulization algorithm failed")

    mol_graph.write_to_smiles_symbols()


# Aromatic Helper Methods and Classes

# key = aromatic SMILES element, value = number of valence electrons
# Note: wild card '*' not supported currently
_aromatic_valences = {
    'b': 3,
    'al': 3,
    'c': 4,
    'si': 4,
    'n': 5,
    'p': 5,
    'as': 5,
    'o': 6,
    's': 6,
    'se': 6,
    'te': 6,
}


def _capitalize(atom_symbol: str) -> str:
    """Capitalizes the element portion of an aromatic SMILES atom symbol,
    converting it into a standard SMILES atom symbol.
    :param atom_symbol: an aromatic SMILES atom symbol.
    :return: the capitalized ``atom_symbol``.
    """

    s, _ = find_element(atom_symbol)
    return atom_symbol[:s] + atom_symbol[s].upper() + atom_symbol[s + 1 :]


def _is_aromatic(atom_symbol: str) -> bool:
    """Checks whether a SMILES atom symbol is an aromatic SMILES atom symbol.
    An aromatic SMILES atom symbol is indicated by an element substring
    that is not capitalized.
    :param atom_symbol: a SMILES atom symbol.
    :return: True, if ``atom_symbol`` is an aromatic atom symbol,
        and False otherwise.
    """

    s, e = find_element(atom_symbol)

    if e == len(atom_symbol):  # optimization to prevent string copying
        element = atom_symbol
    else:
        element = atom_symbol[s:e]

    if element[0].isupper():  # check if element is capitalized
        return False

    if element not in _aromatic_valences:
        raise ValueError("unrecognized aromatic symbol '{}'".format(atom_symbol))
    return True


def _in_pi_subgraph(atom_symbol: str, bonds: Tuple[str]) -> bool:
    """Checks whether a SMILES atom symbol should be a node in the pi
    subgraph, based on its bonds.
    More specifically, an atom should be a node in the pi subgraph if it has
    an unpaired valence electron, and thus, is able to make a double bond.
    Reference: https://depth-first.com/articles/2020/02/10/a-comprehensive
               -treatment-of-aromaticity-in-the-smiles-language/
    :param atom_symbol: a SMILES atom symbol representing an atom.
    :param bonds: the bonds connected to ``atom_symbol``.
    :return: True if ``atom_symbol`` should be included in the pi subgraph,
        and False otherwise.
    """

    atom, h_count, charge = parse_atom_symbol(atom_symbol)

    used_electrons = 0
    for b in bonds:
        used_electrons += get_num_from_bond(b)

    # e.g. c1ccccc1
    # this also covers the neutral carbon radical case (e.g. C1=[C]NC=C1),
    # which is treated equivalently to a 1-H carbon (e.g. C1=[CH]NC=C1)
    if (
        (atom == 'c')
        and (h_count == charge == 0)
        and (len(bonds) == 2)
        and ('#' not in bonds)
    ):

        h_count += 1  # implied bonded hydrogen

    if h_count > 1:
        raise ValueError("unrecognized aromatic symbol '{}'".format(atom_symbol))

    elif h_count == 1:  # e.g. [nH]
        used_electrons += 1

    valence = _aromatic_valences[atom] - charge
    free_electrons = valence - used_electrons
    return free_electrons % 2 != 0


class MolecularGraph:
    """A molecular graph.
    This molecular graph operates based on the ``smiles_symbols`` data
    structure. Indices from this list represent nodes or edges, depending
    on whether they point to a SMILES atom(s) or bond.
    :ivar smiles_symbols: the list created from the iterator returned by
        ``encoder._parse_smiles``. Serves as the base data structure
        of this class, as everything is communicated through indices
        referring to elements of this list.
    :ivar graph: the key is an index of the atom(s) from ``smiles_symbols``.
        The value is a list of Bond objects representing the connected
        bonds. Represents the actual molecular graph.
    :ivar aro_indices: a set of indices of atom(s) from ``smiles_symbols``
        that are aromatic in the molecular graph.
    """

    def __init__(self, smiles_symbols: List[List[Union[str, int]]]):
        self.smiles_symbols = smiles_symbols
        self.graph = {}
        self.aro_indices = set()

    def get_atom_symbol(self, idx: int) -> str:
        """Getter that returns the SMILES symbol representing an atom
        at a specified index.
        :param idx: an index in ``smiles_symbols``.
        :return: the SMILES symbol representing an atom at index
            ``idx`` in ``smiles_symbols``.
        """

        return self.smiles_symbols[idx][1]

    def get_bond_symbol(self, idx: int) -> str:
        """Getter that returns the SMILES symbol representing a bond at
        a specified index.
        :param idx: an index in ``smiles_symbols``.
        :return: the SMILES symbol representing a bond at index
            ``idx`` in ``smiles_symbols``.
        """

        return self.smiles_symbols[idx][0]

    def get_nodes_by_num_edges(self) -> List[int]:
        """Returns all nodes (or indices) stored in this molecular graph
        in a semi-sorted order by number of edges.
        This is to optimize the speed of ``dfs_assign_bonds``; starting
        with nodes that have fewer edges will improve computational time
        as there are fewer bond configurations to explore. Instead of fully
        sorting the returned list, a compromise is made, and nodes with exactly
        one edge are added to the list's beginning.
        :return: a list of the nodes (or indices) of this molecular graph,
            semi-sorted by number of edges.
        """

        ends = []  # nodes with exactly 1 edge
        middles = []  # nodes with 2+ edges

        for idx, edges in self.graph.items():
            if len(edges) > 1:
                middles.append(idx)
            else:
                ends.append(idx)

        ends.extend(middles)
        return ends

    def set_atom_symbol(self, atom_symbol: str, idx: int) -> None:
        """Setter that updates the SMILES symbol representing an atom(s) at
        a specified index.
        :param atom_symbol: the new value of the atom symbol at ``idx``.
        :param idx: an index in ``smiles_symbols``.
        :return: None.
        """

        self.smiles_symbols[idx][1] = atom_symbol

    def set_bond_symbol(self, bond_symbol: str, idx: int) -> None:
        """Setter that updates the SMILES symbol representing a bond at
        a specified index.
        :param bond_symbol: the new value of the bond symbol at ``idx``.
        :param idx: an index in ``smiles_symbols``.
        :return: None.
        """

        self.smiles_symbols[idx][0] = bond_symbol

    def add_bond(self, idx_a: int, idx_b: int, bond_idx: int) -> None:
        """Adds a bond (or edge) to this molecular graph between atoms
        (or nodes) at two specified indices.
        :param idx_a: the index of one atom (or node) of this bond.
        :param idx_b:the index of one atom (or node) of this bond.
        :param bond_idx: the index of this bond.
        :return: None.
        """

        atom_a = self.get_atom_symbol(idx_a)
        atom_b = self.get_atom_symbol(idx_b)
        atom_a_aro = (idx_a in self.aro_indices) or _is_aromatic(atom_a)
        atom_b_aro = (idx_b in self.aro_indices) or _is_aromatic(atom_b)
        bond_symbol = self.get_bond_symbol(bond_idx)

        if atom_a_aro:
            self.aro_indices.add(idx_a)

        if atom_b_aro:
            self.aro_indices.add(idx_b)

        if bond_symbol == ':':
            self.aro_indices.add(idx_a)
            self.aro_indices.add(idx_b)

            # Note: ':' bonds are edited here to ''
            self.set_bond_symbol('', bond_idx)
            bond_symbol = ''

        edge = Bond(idx_a, idx_b, bond_symbol, bond_idx)

        self.graph.setdefault(idx_a, []).append(edge)
        self.graph.setdefault(idx_b, []).append(edge)

    def prune_to_pi_subgraph(self) -> None:
        """Removes nodes and edges from this molecular graph such that
        it becomes the pi subgraph.
        The remaining graph will only contain aromatic atoms (or nodes)
        that belong in the pi-subgraph, and the bonds that are aromatic
        and between such atoms.
        :return: None.
        """

        # remove non-aromatic nodes
        non_aromatic = self.graph.keys() - self.aro_indices
        for i in non_aromatic:
            self.graph.pop(i)

        # remove non-pi subgraph nodes
        for i in self.aro_indices:

            atom = self.get_atom_symbol(i)
            bonds = tuple(edge.bond_symbol for edge in self.graph[i])

            if not _in_pi_subgraph(atom, bonds):
                self.graph.pop(i)

        # remove irrelevant edges
        for idx, edges in self.graph.items():

            keep = list(
                filter(
                    lambda e: (e.idx_a in self.graph)
                    and (e.idx_b in self.graph)
                    and (e.bond_symbol == ''),
                    edges,
                )
            )
            self.graph[idx] = keep

    def dfs_assign_bonds(
        self, idx: int, visited: Set[int], matched_nodes: Set[int], matched_edges
    ) -> bool:
        """After calling ``prune_to_pi_subgraph``, this method assigns
        double bonds between pairs of nodes such that every node is
        paired or matched.
        This is done recursively in a depth-first search fashion.
        :param idx: the index of the current atom (or node).
        :param visited: a set of the indices of nodes that have been visited.
        :param matched_nodes: a set of the indices of nodes that have been
            matched, i.e., assigned a double bond.
        :param matched_edges: a set of the bonds that have been matched.
        :return: True, if a valid bond assignment was found; False otherwise.
        """

        if idx in visited:
            return True

        edges = self.graph[idx]

        if idx in matched_nodes:

            # recursively try to match adjacent nodes. If the matching
            # fails, then we must backtrack.
            visited_save = visited.copy()

            visited.add(idx)
            for e in edges:
                adj = e.other_end(idx)
                if not self.dfs_assign_bonds(
                    adj, visited, matched_nodes, matched_edges
                ):
                    visited &= visited_save
                    return False
            return True

        else:

            # list of candidate edges that can become a double bond
            candidates = list(
                filter(lambda i: i.other_end(idx) not in matched_nodes, edges)
            )

            if not candidates:
                return False  # idx is unmatched, but all adj nodes are matched

            matched_edges_save = matched_edges.copy()

            for e in candidates:

                # match nodes connected by c
                matched_nodes.add(e.idx_a)
                matched_nodes.add(e.idx_b)
                matched_edges.add(e)

                success = self.dfs_assign_bonds(
                    idx, visited, matched_nodes, matched_edges
                )

                if success:
                    e.bond_symbol = '='
                    return True
                else:  # the matching failed, so we must backtrack

                    for edge in matched_edges - matched_edges_save:
                        edge.bond_symbol = ''
                        matched_nodes.discard(edge.idx_a)
                        matched_nodes.discard(edge.idx_b)

                    matched_edges &= matched_edges_save

            return False

    def write_to_smiles_symbols(self):
        """Updates and mutates ``self.smiles_symbols`` with the information
         contained in ``self.graph``.
        After kekulizing the molecular graph, this method is called to
        merge the new information back into the original data structure.
        :return: None.
        """

        # capitalize aromatic molecules
        for idx in self.aro_indices:
            self.set_atom_symbol(_capitalize(self.get_atom_symbol(idx)), idx)

        # write bonds
        for edge_list in self.graph.values():
            for edge in edge_list:
                bond_symbol = edge.bond_symbol
                bond_idx = edge.bond_idx

                self.set_bond_symbol(bond_symbol, bond_idx)

                # branches record the next symbol as their bond, so we
                # must update accordingly
                if (bond_idx > 0) and (
                    self.smiles_symbols[bond_idx - 1][2] == BRANCH_TYPE
                ):
                    self.set_bond_symbol(bond_symbol, bond_idx - 1)


class Bond:
    """Represents a bond or edge in MolecularGraph.
    Recall that the following indices are with respect to ``smiles_symbols``
    in MolecularGraph.
    :ivar idx_a: the index of one atom or node of this bond.
    :ivar idx_b: the index of one atom or node of this bond.
    :ivar bond_symbol: the SMILES symbol representing this bond (e.g. '#').
    :ivar bond_idx: the index of this bond or edge.
    """

    def __init__(self, idx_a, idx_b, bond_symbol, bond_idx):
        self.idx_a = idx_a
        self.idx_b = idx_b
        self.bond_symbol = bond_symbol
        self.bond_idx = bond_idx

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return (self.idx_a, self.idx_b) == (other.idx_a, other.idx_b)
        return NotImplemented

    def __hash__(self):
        return hash((self.idx_a, self.idx_b))

    def other_end(self, idx):
        """Given an index representing one end of this bond, returns
        the index representing the other end.
        :param idx: an index of one atom or node of this bond.
        :return: the index of the other atom or node of this bond, or
            None if ``idx`` is an invalid input.
        """

        if idx == self.idx_a:
            return self.idx_b
        elif idx == self.idx_b:
            return self.idx_a
        return None


############### Decoder ####################
def decoder(
    selfies: str, print_error: bool = False, constraints: Optional[str] = None
) -> Optional[str]:
    """Translates a SELFIES into a SMILES.
    The SELFIES to SMILES translation operates based on the :mod:`selfies`
    grammar rules, which can be configured using
    :func:`selfies.set_semantic_constraints`. Given the appropriate settings,
    the decoded SMILES will always be syntactically and semantically correct.
    That is, the output SMILES will satisfy the specified bond constraints.
    Additionally, :func:`selfies.decoder` will attempt to preserve the
    atom and branch order of the input SELFIES.
    :param selfies: the SELFIES to be translated.
    :param print_error: if True, error messages will be printed to console.
        Defaults to False.
    :param constraints: if ``'octet_rule'`` or ``'hypervalent'``,
        the corresponding preset bond constraints will be used instead.
        If ``None``, :func:`selfies.decoder` will use the
        currently configured bond constraints. Defaults to ``None``.
    :return: the SMILES translation of ``selfies``. If an error occurs,
        and ``selfies`` cannot be translated, ``None`` is returned instead.
    :Example:
    >>> import selfies
    >>> selfies.decoder('[C][=C][F]')
    'C=CF'
    .. seealso:: The
        `"octet_rule" <https://en.wikipedia.org/wiki/Octet_rule>`_
        and
        `"hypervalent" <https://en.wikipedia.org/wiki/Hypervalent_molecule>`_
        preset bond constraints
        can be viewed with :func:`selfies.get_octet_rule_constraints` and
        :func:`selfies.get_hypervalent_constraints`, respectively. These
        presets are variants of the "default" bond constraints, which can
        be viewed with :func:`selfies.get_default_constraints`. Their
        differences can be summarized as follows:
            * def. : ``Cl``, ``Br``, ``I``: 1, ``N``: 3, ``P``: 5, ``P+1``: 6, ``P-1``: 4, ``S``: 6, ``S+1``: 7, ``S-1``: 5
            * oct. : ``Cl``, ``Br``, ``I``: 1, ``N``: 3, ``P``: 3, ``P+1``: 4, ``P-1``: 2, ``S``: 2, ``S+1``: 3, ``S-1``: 1
            * hyp. : ``Cl``, ``Br``, ``I``: 7, ``N``: 5, ``P``: 5, ``P+1``: 6, ``P-1``: 4, ``S``: 6, ``S+1``: 7, ``S-1``: 5
    """

    old_constraints = get_semantic_constraints()
    if constraints is None:
        pass
    elif constraints == 'octet_rule':
        set_semantic_constraints(get_octet_rule_constraints())
    elif constraints == 'hypervalent':
        set_semantic_constraints(get_hypervalent_constraints())
    else:
        raise ValueError("unrecognized constraint type")

    try:
        all_smiles = []  # process dot-separated fragments separately

        for s in selfies.split("."):
            smiles = _translate_selfies(s)

            if smiles != "":  # prevent malformed dots (e.g. [C]..[C], .[C][C])
                all_smiles.append(smiles)

        if constraints is not None:  # restore old constraints
            set_semantic_constraints(old_constraints)

        return '.'.join(all_smiles)

    except ValueError as err:
        if constraints is not None:  # restore old constraints
            set_semantic_constraints(old_constraints)

        if print_error:
            print("Decoding error '{}': {}.".format(selfies, err))
        return None


def _parse_selfies(selfies: str) -> Iterable[str]:
    """Parses a SELFIES into its symbols.
    A generator, which parses a SELFIES and yields its symbols
    one-by-one. When no symbols are left in the SELFIES, the empty
    string is infinitely yielded. As a precondition, the input SELFIES contains
    no dots, so all symbols are enclosed by square brackets, e.g. [X].
    :param selfies: the SElFIES string to be parsed.
    :return: an iterable of the symbols of the SELFIES.
    """

    left_idx = selfies.find('[')

    while 0 <= left_idx < len(selfies):
        right_idx = selfies.find(']', left_idx + 1)

        if (selfies[left_idx] != '[') or (right_idx == -1):
            raise ValueError("malformed SELIFES, " "misplaced or missing brackets")

        next_symbol = selfies[left_idx : right_idx + 1]
        left_idx = right_idx + 1

        if next_symbol != '[nop]':  # skip [nop]
            yield next_symbol

    while True:  # no more symbols left
        yield ''


def _parse_selfies_symbols(selfies_symbols: List[str]) -> Iterable[str]:
    """Equivalent to ``_parse_selfies``, except the input SELFIES is presented
    as a list of SELFIES symbols, as opposed to a string.
    :param selfies_symbols: a SELFIES represented as a list of SELFIES symbols.
    :return: an iterable of the symbols of the SELFIES.
    """
    for symbol in selfies_symbols:

        if symbol != '[nop]':
            yield symbol

    while True:
        yield ''


def _translate_selfies(selfies: str) -> str:
    """A helper for ``selfies.decoder``, which translates a SELFIES into a
    SMILES (assuming the input SELFIES contains no dots).
    :param selfies: the SELFIES to be translated.
    :return: the SMILES translation of the SELFIES.
    """

    selfies_gen = _parse_selfies(selfies)

    # derived[i] is a list with three elements:
    #  (1) a string representing the i-th derived atom, and its connecting
    #      bond (e.g. =C, #N, N, C are all possible)
    #  (2) the number of available bonds the i-th atom has to make
    #  (3) the index of the previously derived atom that the i-th derived
    #      atom is bonded to
    # Example: if the 6-th derived atom was 'C', had 2 available bonds,
    # and was connected to the 5-th derived atom by a double bond, then
    # derived[6] = ['=C', 2, 5]
    derived = []

    # each item of <branches> is a key-value pair of indices that represents
    # the branches to be made. If a branch starts at the i-th derived atom
    # and ends at the j-th derived atom, then branches[i] = j. No two
    # branches should start at the same atom, e.g. C((C)Cl)C
    branches = {}

    # each element of <rings> is a tuple of size three that represents the
    # rings to be made, in the same order they appear in the SELFIES (left
    # to right). If the i-th ring is between the j-th and k-th derived atoms
    # (j <= k) and has bond symbol s ('=', '#', '\', etc.), then
    # rings[i] = (j, k, s).
    rings = []

    _translate_selfies_derive(selfies_gen, 0, derived, -1, branches, rings)
    _form_rings_bilocally(derived, rings)

    # create branches
    for lb, rb in branches.items():
        derived[lb][0] = '(' + derived[lb][0]
        derived[rb][0] += ')'

    smiles = ""
    for s, _, _ in derived:  # construct SMILES from <derived>
        smiles += s
    return smiles


# flake8: noqa: C901
# noinspection PyTypeChecker
def _translate_selfies_derive(
    selfies_gen: Iterable[str],
    init_state: int,
    derived: List[List[Union[str, int]]],
    prev_idx: int,
    branches: Dict[int, int],
    rings: List[Tuple[int, int, str]],
) -> None:
    """Recursive helper for _translate_selfies.
    Derives the SMILES symbols one-by-one from a SELFIES, and
    populates derived, branches, and rings. The main chain and side branches
    of the SELFIES are translated recursively. Rings are not actually
    translated, but saved to the rings list to be added later.
    :param selfies_gen: an iterable of the symbols of the SELFIES to be
        translated, created by ``_parse_selfies``.
    :param init_state: the initial derivation state.
    :param derived: see ``derived`` in ``_translate_selfies``.
    :param prev_idx: the index of the previously derived atom, or -1,
        if no atoms have been derived yet.
    :param branches: see ``branches`` in ``_translate_selfies``.
    :param rings: see ``rings`` in ``_translate_selfies``.
    :return: ``None``.
    """

    curr_symbol = next(selfies_gen)
    state = init_state

    while curr_symbol != '' and state >= 0:

        # Case 1: Branch symbol (e.g. [Branch1_2])
        if 'Branch' in curr_symbol:

            branch_init_state, new_state = get_next_branch_state(curr_symbol, state)

            if state <= 1:  # state = 0, 1
                pass  # ignore no symbols

            else:
                L = int(curr_symbol[-4])  # corresponds to [BranchL_X]
                L_symbols = []
                for _ in range(L):
                    L_symbols.append(next(selfies_gen))

                N = get_n_from_symbols(*L_symbols)

                branch_symbols = []
                for _ in range(N + 1):
                    branch_symbols.append(next(selfies_gen))
                branch_gen = _parse_selfies_symbols(branch_symbols)

                branch_start = len(derived)
                _translate_selfies_derive(
                    branch_gen, branch_init_state, derived, prev_idx, branches, rings
                )
                branch_end = len(derived) - 1

                # resolve C((C)Cl)C --> C(C)(Cl)C
                while branch_start in branches:
                    branch_start = branches[branch_start] + 1

                # finally, register the branch in branches
                if branch_start <= branch_end:
                    branches[branch_start] = branch_end

        # Case 2: Ring symbol (e.g. [Ring2])
        elif 'Ring' in curr_symbol:

            new_state = state

            if state == 0:
                pass  # ignore no symbols

            else:
                L = int(curr_symbol[-2])  # corresponds to [RingL]
                L_symbols = []
                for _ in range(L):
                    L_symbols.append(next(selfies_gen))

                N = get_n_from_symbols(*L_symbols)

                left_idx = max(0, prev_idx - (N + 1))
                right_idx = prev_idx

                bond_symbol = ''
                if curr_symbol[1:5] == 'Expl':
                    bond_symbol = curr_symbol[5]

                rings.append((left_idx, right_idx, bond_symbol))

        # Case 3: regular symbol (e.g. [N], [=C], [F])
        else:
            new_symbol, new_state = get_next_state(curr_symbol, state)

            if new_symbol != '':  # in case of [epsilon]
                derived.append([new_symbol, new_state, prev_idx])

                if prev_idx >= 0:
                    bond_num = get_num_from_bond(new_symbol[0])
                    derived[prev_idx][1] -= bond_num

                prev_idx = len(derived) - 1

        curr_symbol = next(selfies_gen)  # update symbol and state
        state = new_state


def _form_rings_bilocally(
    derived: List[List[Union[str, int]]], rings: List[Tuple[int, int, str]]
) -> None:
    """Forms all the rings specified by the rings list, in first-to-last order,
    by updating derived.
    :param derived: see ``derived`` in ``_translate_selfies``.
    :param rings: see ``rings`` in ``_translate_selfies``.
    :return: ``None``.
    """

    # due to the behaviour of allowing multiple rings between the same atom
    # pair, or rings between already bonded atoms, we first resolve all rings
    # so that only valid rings are left and placed into <ring_locs>.
    ring_locs = OrderedDict()

    for left_idx, right_idx, bond_symbol in rings:

        if left_idx == right_idx:  # ring to the same atom forbidden
            continue

        left_end = derived[left_idx]
        right_end = derived[right_idx]
        bond_num = get_num_from_bond(bond_symbol)

        if left_end[1] <= 0 or right_end[1] <= 0:
            continue  # no room for bond

        if bond_num > min(left_end[1], right_end[1]):
            bond_num = min(left_end[1], right_end[1])
            bond_symbol = get_bond_from_num(bond_num)

        # ring is formed between two atoms that are already bonded
        # e.g. CC1C1C --> CC=CC
        if left_idx == right_end[2]:

            right_symbol = right_end[0]

            if right_symbol[0] in {'-', '/', '\\', '=', '#'}:
                old_bond = right_symbol[0]
            else:
                old_bond = ''

            # update bond multiplicity and symbol
            new_bond_num = min(bond_num + get_num_from_bond(old_bond), 3)
            new_bond_symbol = get_bond_from_num(new_bond_num)

            right_end[0] = new_bond_symbol + right_end[0][len(old_bond) :]

        # ring is formed between two atoms that are not bonded, e.g. C1CC1C
        else:
            loc = (left_idx, right_idx)

            if loc in ring_locs:
                # a ring is formed between two atoms that are have previously
                # been bonded by a ring, so ring bond multiplicity is updated

                new_bond_num = min(bond_num + get_num_from_bond(ring_locs[loc]), 3)
                new_bond_symbol = get_bond_from_num(new_bond_num)
                ring_locs[loc] = new_bond_symbol

            else:
                ring_locs[loc] = bond_symbol

        left_end[1] -= bond_num
        right_end[1] -= bond_num

    # finally, use <ring_locs> to add all the rings into <derived>

    ring_counter = 1
    for (left_idx, right_idx), bond_symbol in ring_locs.items():

        ring_id = str(ring_counter)
        if len(ring_id) == 2:
            ring_id = "%" + ring_id
        ring_counter += 1  # increment

        derived[left_idx][0] += bond_symbol + ring_id
        derived[right_idx][0] += bond_symbol + ring_id


def split_selfies(selfies: str) -> Iterable[str]:
    """Splits a SELFIES into its symbols.
    Returns an iterable that yields the symbols of a SELFIES one-by-one
    in the order they appear in the string. SELFIES symbols are always
    either indicated by an open and closed square bracket, or are the ``'.'``
    dot-bond symbol.
    :param selfies: the SELFIES to be read.
    :return: an iterable of the symbols of ``selfies`` in the same order
        they appear in the string.
    :Example:
    >>> import selfies
    >>> list(selfies.split_selfies('[C][O][C]'))
    ['[C]', '[O]', '[C]']
    >>> list(selfies.split_selfies('[C][=C][F].[C]'))
    ['[C]', '[=C]', '[F]', '.', '[C]']
    """

    left_idx = selfies.find("[")

    while 0 <= left_idx < len(selfies):
        right_idx = selfies.find("]", left_idx + 1)
        next_symbol = selfies[left_idx : right_idx + 1]
        yield next_symbol

        left_idx = right_idx + 1
        if selfies[left_idx : left_idx + 1] == ".":
            yield "."
            left_idx += 1


###################### Encoder #######################


def encoder(smiles: str, print_error: bool = False) -> Optional[str]:
    """Translates a SMILES into a SELFIES.
    The SMILES to SELFIES translation occurs independently of the SELFIES
    alphabet and grammar. Thus, :func:`selfies.encoder` will work regardless of
    the alphabet and grammar rules that :py:mod:`selfies` is operating on,
    assuming the input is a valid SMILES. Additionally, :func:`selfies.encoder`
    preserves the atom and branch order of the input SMILES; thus, one
    could generate random SELFIES corresponding to the same molecule by
    generating random SMILES, and then translating them.
    However, encoding and then decoding a SMILES may not necessarily yield
    the original SMILES. Reasons include:
        1.  SMILES with aromatic symbols are automatically
            Kekulized before being translated.
        2.  SMILES that violate the bond constraints specified by
            :mod:`selfies` will be successfully encoded by
            :func:`selfies.encoder`, but then decoded into a new molecule
            that satisfies the constraints.
        3.  The exact ring numbering order is lost in :func:`selfies.encoder`,
            and cannot be reconstructed by :func:`selfies.decoder`.
    Finally, note that :func:`selfies.encoder` does **not** check if the input
    SMILES is valid, and should not be expected to reject invalid inputs.
    It is recommended to use RDKit to first verify that the SMILES are
    valid.
    :param smiles: the SMILES to be translated.
    :param print_error: if True, error messages will be printed to console.
        Defaults to False.
    :return: the SELFIES translation of ``smiles``. If an error occurs,
        and ``smiles`` cannot be translated, :code:`None` is returned instead.
    :Example:
    >>> import selfies
    >>> selfies.encoder('C=CF')
    '[C][=C][F]'
    .. note:: Currently, :func:`selfies.encoder` does not support the
        following types of SMILES:
        *   SMILES using ring numbering across a dot-bond symbol
            to specify bonds, e.g. ``C1.C2.C12`` (propane) or
            ``c1cc([O-].[Na+])ccc1`` (sodium phenoxide).
        *   SMILES with ring numbering between atoms that are over
            ``16 ** 3 = 4096`` atoms apart.
        *   SMILES using the wildcard symbol ``*``.
        *   SMILES using chiral specifications other than ``@`` and ``@@``.
    """

    try:
        if '*' in smiles:
            raise ValueError("wildcard atom '*' not supported")

        all_selfies = []  # process dot-separated fragments separately
        for s in smiles.split("."):
            all_selfies.append(_translate_smiles(s))
        return '.'.join(all_selfies)

    except ValueError as err:
        if print_error:
            print("Encoding error '{}': {}.".format(smiles, err))
        return None


def _translate_smiles(smiles: str) -> str:
    """A helper for ``selfies.encoder``, which translates a SMILES into a
    SELFIES (assuming the input SMILES contains no dots).
    :param smiles: the SMILES to be translated.
    :return: the SELFIES translation of SMILES.
    """

    smiles_gen = _parse_smiles(smiles)

    char_set = set(smiles)
    if any(c in char_set for c in ['c', 'n', 'o', 'p', 'a', 's']):
        smiles_gen = kekulize_parser(smiles_gen)

    # a simple mutable counter to track which atom was the i-th derived atom
    derive_counter = [0]

    # a dictionary to keep track of the rings to be made. If a ring with id
    # X is connected to the i-th and j-th derived atoms (i < j) with bond
    # symbol s, then after the i-th atom is derived, rings[X] = (s, i).
    # As soon as the j-th atom is derived, rings[X] is removed from <rings>,
    # and the ring is made.
    rings = {}

    selfies, _ = _translate_smiles_derive(smiles_gen, rings, derive_counter)

    if rings:
        raise ValueError(
            "malformed ring numbering or ring numbering " "across a dot symbol"
        )

    return selfies


def _parse_smiles(smiles: str) -> Iterable[Tuple[str, str, int]]:
    """Parses a SMILES into its symbols.
    A generator, which parses a SMILES string and returns its symbol(s)
    one-by-one as a tuple of:
        (1) the bond symbol connecting the current atom/ring/branch symbol
            to the previous atom/ring/branch symbol (e.g. '=', '', '#')
        (2) the atom/ring/branch symbol as a string (e.g. 'C', '12', '(')
        (3) the type of the symbol in (2), represented as an integer that is
            either ``ATOM_TYPE``, ``BRANCH_TYPE``, and ``RING_TYPE``.
    As a precondition, we also assume ``smiles`` has no dots in it.
    :param smiles: the SMILES to be parsed.
    :return: an iterable of the symbol(s) of the SELFIES along with
        their types.
    """

    i = 0

    while 0 <= i < len(smiles):

        bond = ''

        if smiles[i] in {'-', '/', '\\', '=', '#', ":"}:
            bond = smiles[i]
            i += 1

        if smiles[i].isalpha():  # organic subset elements
            if smiles[i : i + 2] in ('Br', 'Cl'):  # two letter elements
                symbol = smiles[i : i + 2]
                symbol_type = ATOM_TYPE
                i += 2
            else:
                symbol = smiles[i]  # one letter elements (e.g. C, N, ...)
                symbol_type = ATOM_TYPE
                i += 1

        elif smiles[i] in ('(', ')'):  # open and closed branch brackets
            bond = smiles[i + 1 : i + 2]
            symbol = smiles[i]
            symbol_type = BRANCH_TYPE
            i += 1

        elif smiles[i] == '[':  # atoms encased in brackets (e.g. [NH])
            r_idx = smiles.find(']', i + 1)
            symbol = smiles[i : r_idx + 1]
            symbol_type = ATOM_TYPE
            i = r_idx + 1

            if r_idx == -1:
                raise ValueError("malformed SMILES, missing ']'")

            # quick chirality specification check
            chiral_i = symbol.find('@')
            if symbol[chiral_i + 1].isalpha() and symbol[chiral_i + 1] != 'H':
                raise ValueError(
                    "chiral specification '{}' not supported".format(symbol)
                )

        elif smiles[i].isdigit():  # one-digit ring number
            symbol = smiles[i]
            symbol_type = RING_TYPE
            i += 1

        elif smiles[i] == '%':  # two-digit ring number (e.g. %12)
            symbol = smiles[i + 1 : i + 3]
            symbol_type = RING_TYPE
            i += 3

        else:
            raise ValueError("unrecognized symbol '{}'".format(smiles[i]))

        yield bond, symbol, symbol_type


def _translate_smiles_derive(
    smiles_gen: Iterable[Tuple[str, str, int]],
    rings: Dict[int, Tuple[str, int]],
    counter: List[int],
) -> Tuple[str, int]:
    """Recursive helper for _translate_smiles.
    Derives the SELFIES from a SMILES, and returns a tuple of (1) the
    translated SELFIES and (2) the symbol length of the translated SELFIES.
    :param smiles_gen: an iterable of the symbols (and their types)
        of the SMILES to be translated, created by ``_parse_smiles``.
    :param rings: See ``rings`` in ``_translate_smiles``.
    :param counter: a one-element list that serves as a mutable counter.
        See ``derived_counter`` in ``_translate_smiles``.
    :return: A tuple of the translated SELFIES and its symbol length.
    """

    selfies = ""
    selfies_len = 0
    prev_idx = -1

    for bond, symbol, symbol_type in smiles_gen:

        if bond == '-':  # ignore explicit single bonds
            bond = ''

        if symbol_type == ATOM_TYPE:
            if symbol[0] == '[':
                selfies += "[{}{}expl]".format(bond, symbol[1:-1])
            else:
                selfies += "[{}{}]".format(bond, symbol)
            prev_idx = counter[0]
            counter[0] += 1
            selfies_len += 1

        elif symbol_type == BRANCH_TYPE:
            if symbol == '(':

                # NOTE: looping inside a loop on a generator will produce
                # expected behaviour in this case.

                branch, branch_len = _translate_smiles_derive(
                    smiles_gen, rings, counter
                )

                N_as_symbols = get_symbols_from_n(branch_len - 1)
                bond_num = get_num_from_bond(bond)

                selfies += "[Branch{}_{}]".format(len(N_as_symbols), bond_num)
                selfies += ''.join(N_as_symbols) + branch
                selfies_len += 1 + len(N_as_symbols) + branch_len

            else:  # symbol == ')'
                break

        else:  # symbol_type == RING_TYPE
            ring_id = int(symbol)

            if ring_id in rings:
                left_bond, left_end = rings.pop(ring_id)
                right_bond, right_end = bond, prev_idx

                ring_len = right_end - left_end
                N_as_symbols = get_symbols_from_n(ring_len - 1)

                if left_bond != '':
                    selfies += "[Expl{}Ring{}]".format(left_bond, len(N_as_symbols))
                elif right_bond != '':
                    selfies += "[Expl{}Ring{}]".format(right_bond, len(N_as_symbols))
                else:
                    selfies += "[Ring{}]".format(len(N_as_symbols))

                selfies += ''.join(N_as_symbols)
                selfies_len += 1 + len(N_as_symbols)

            else:
                rings[ring_id] = (bond, prev_idx)

    return selfies, selfies_len

import pytest
import nvc_toolkit as nvc
from typing import Tuple
import pandas as pd

# TODO - refactor using decorators and class structure to rmv duplicate code

inp_test_data = [
    "when you fight with me I feel angry",
    "I hate when you do that shit"
]

# DATA LOADING/CLEARNING/PARSING FUNCS #
@pytest.fixture
def kwp_dfs():
    return nvc.load_dfs()

def test_load_dfs(kwp_dfs:Tuple):
    # I know, multiple asserts in 1 fctn bad bad...idcrn
    assert isinstance(kwp_dfs, Tuple)
    assert isinstance(kwp_dfs[0], pd.DataFrame)
    assert isinstance(kwp_dfs[1], pd.DataFrame)

def test_cleaning(kwp_dfs:Tuple):
    assert isinstance(nvc.clean_df(kwp_dfs[0]), pd.DataFrame)
    assert isinstance(nvc.clean_df(kwp_dfs[1]), pd.DataFrame)

# UTILITY FUNCS #
@pytest.mark.parametrize("inp", inp_test_data)
def test_parse_sent(inp:str):
    parsed = nvc.parse_sent(inp)
    assert isinstance(parsed, pd.DataFrame)

@pytest.mark.parametrize("inp", inp_test_data)
def test_find_kwp_matches(inp:str, kwp_dfs:Tuple):
    parsed = nvc.parse_sent(inp)
    kwp_matches = nvc.find_kwp_matches(parsed, kwp_dfs)
    assert isinstance(kwp_matches, pd.Series)

@pytest.mark.parametrize("inp", inp_test_data)
def test_find_pos_matches(inp:str):
    parsed = nvc.parse_sent(inp)
    pos_matches = nvc.find_pos_matches(parsed)
    assert isinstance(pos_matches, pd.Series)


# # USER FUNCS
# class TestUserFuncs:
    # @pytest.mark.parametrize("earned,spent,expected", [
    #     (30, 10, 20),
    #     (20, 2, 18),
    # ])
    # def test_transactions(earned, spent, expected):
    #     my_wallet = Wallet()
    #     my_wallet.add_cash(earned)
    #     my_wallet.spend_cash(spent)
    #     assert my_wallet.balance == expected
    # def test_get_raw_feedback(self):
    #     pass
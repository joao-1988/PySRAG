'''
This script is based on the code from this link: https://github.com/FluVigilanciaBR/fludashboard/blob/master/Notebooks/episem.py

Brazilian epidemiological week (epiweek) is definied from Sunday to Saturday. 
Wednesday is defined as the turning point for changing from one year to another and deciding wether Jan 1st is inlcuded in the first epiweek of the new year or still in the last epiweek of the previous one.
That is, if the weekday of Jan 1st is between Sunday to Wednesday (included), then it is epiweek 01 of the new year. If the weekday is between Thursday-Saturday, then it falls in the last epiweek of the previous year (typically epiweek 52).
'''

from datetime import datetime, timedelta

__all__ = ["EpiWeek"]


class EpiWeek:
    @staticmethod
    def weekday(date):
        """
        Extract weekday from date.
        :param date: datetime.date
        :return: int [0-6] (Sun-Sat)
        """
        return int(date.strftime('%w'))

    @staticmethod
    def firstday_epiyear(year):
        """
        Get the first day of the first epidemiological week of the year.
        :param year: int
        :return: datetime.date
        """
        date = datetime(year, 1, 1)
        weekday = EpiWeek.weekday(date)

        if weekday <= 3:
            date -= timedelta(days=weekday)
        else:
            date += timedelta(days=7 - weekday)
        return date

    @staticmethod
    def lastday_epiyear(year):
        """
        Get the last day of the last epidemiological week of the year.
        :param year: int
        :return: datetime.date
        """
        return EpiWeek.firstday_epiyear(year + 1) - timedelta(days=1)

    @staticmethod
    def epiweek(date):
        """
        Get the epidemiological week from a date.
        :param date: datetime.date
        :return: dict {'year': int, 'week': int, 'epiweek': int}
        """
        year = date.year
        firstday = EpiWeek.firstday_epiyear(year)
        lastday = EpiWeek.lastday_epiyear(year)

        if date < firstday:
            year -= 1
            firstday = EpiWeek.firstday_epiyear(year)
        elif date > lastday:
            year += 1
            firstday = EpiWeek.firstday_epiyear(year)

        week = int((date - firstday).days / 7) + 1
        epiweek = year * 100 + week

        return {'year': year, 'week': week, 'epiweek': epiweek}

    @staticmethod
    def epidate(year, week):
        """
        Get the date corresponding to the first day of the epidemiological week.
        :param year: int
        :param week: int
        :return: datetime.date
        """
        firstday = EpiWeek.firstday_epiyear(year)
        return firstday + timedelta(days=(week - 1) * 7)